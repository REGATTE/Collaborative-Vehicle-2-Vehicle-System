import carla
import random
import pygame
import numpy as np
import logging
import argparse

from agents.controller import ControlObject
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, cleanup
from Simulation.sensors import Sensors  # For managing sensor attachments
from agents.EnvironmentManager import EnvironmentManager
from agents.visualize_ego import visualize_ego_sensors
from utils.config.config_loader import load_config


def initialize_carla():
    """
    Initializes the CARLA client and connects to the server.
    :return: client instance and the CARLA world object.
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


def setup_synchronous_mode(world, traffic_manager, config):
    """
    Configures the CARLA server and traffic manager to run in synchronous mode.
    :param world: The CARLA world object.
    :param traffic_manager: The CARLA traffic manager instance.
    :param config: Configuration object for simulation settings.
    """
    settings = world.get_settings()
    settings.synchronous_mode = config.simulation.synchronous_mode
    settings.fixed_delta_seconds = config.simulation.fixed_delta_seconds
    world.apply_settings(settings)

    # Set traffic manager behavior
    traffic_manager.set_global_distance_to_leading_vehicle(5.0)  # Maintain safe distance
    traffic_manager.set_synchronous_mode(config.simulation.synchronous_mode)
    traffic_manager.set_random_device_seed(0)  # Consistent behavior
    random.seed(0)


class RenderObject:
    """
    Handles rendering images from the camera to a PyGame surface.
    """
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


def attach_sensor_suite(world, vehicle, sensor_suite):
    """
    Attaches a suite of sensors to a vehicle in the CARLA world.
    :param world: The CARLA world object.
    :param vehicle: The vehicle to which sensors are attached.
    :param sensor_suite: List of sensor specifications.
    :return: List of spawned sensor actors.
    """
    sensors = []
    for sensor_spec in sensor_suite:
        # Set up sensor blueprint and transform
        sensor_bp = world.get_blueprint_library().find(sensor_spec['type'])
        transform = carla.Transform(
            carla.Location(
                x=sensor_spec.get('x', 0.0),
                y=sensor_spec.get('y', 0.0),
                z=sensor_spec.get('z', 0.0)
            ),
            carla.Rotation(
                pitch=sensor_spec.get('pitch', 0.0),
                yaw=sensor_spec.get('yaw', 0.0),
                roll=sensor_spec.get('roll', 0.0)
            )
        )
        # Spawn the sensor
        sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        # Set additional attributes if specified
        for key, value in sensor_spec.items():
            if sensor_bp.has_attribute(key):
                sensor_bp.set_attribute(key, str(value))

        sensors.append(sensor)
        logging.info(f"Attached sensor {sensor_spec['type']} to vehicle {vehicle.id}.")
    return sensors


def initialize_pygame(camera_bp):
    """
    Initializes the PyGame window based on the camera settings.
    :param camera_bp: Blueprint of the camera.
    :return: PyGame display surface and camera resolution (width, height).
    """
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    pygame.init()
    game_display = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    game_display.fill((0, 0, 0))
    pygame.display.flip()

    return game_display, image_w, image_h


def pygame_callback(data, obj):
    """
    Processes camera data and updates the PyGame display surface.
    :param data: Image data from the camera.
    :param obj: RenderObject to update the surface.
    """
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]  # Extract RGB channels
    img = img[:, :, ::-1]  # Convert from BGRA to RGB
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


def attach_camera(world, ego_vehicle, camera_transform):
    """
    Attaches a camera to the ego vehicle.
    :param world: The CARLA world object.
    :param ego_vehicle: The ego vehicle actor.
    :param camera_transform: Transform to position the camera relative to the vehicle.
    :return: The spawned camera actor and its blueprint.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    return camera, camera_bp

def game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform, vehicle_mapping, intrinsic_matrix, env_manager):
    """
    Main game loop for updating the CARLA world and PyGame display.
    Allows switching between vehicles using the Tab key.
    """
    crashed = False
    font = pygame.font.SysFont('Arial', 16)
    width, height = game_display.get_size()

    # List of all vehicles
    vehicle_keys = list(vehicle_mapping.keys())
    current_vehicle_index = 0  # Start with the first vehicle (typically ego)

    while not crashed:
        world.tick()  # Advance simulation by one tick
        game_display.blit(render_object.surface, (0, 0))

        # Render vehicle labels using the env_manager instance
        env_manager.draw_vehicle_labels(game_display, font, camera, vehicle_mapping, intrinsic_matrix, width, height)

        pygame.display.flip()
        control_object.process_control()

        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    # Switch to the next vehicle
                    current_vehicle_index = (current_vehicle_index + 1) % len(vehicle_keys)
                    current_vehicle_label = vehicle_keys[current_vehicle_index]
                    logging.info(f"Switched to vehicle: {current_vehicle_label}")

                    # Attach camera to the newly selected vehicle
                    new_vehicle = vehicle_mapping[current_vehicle_label]["actor"]

                    # Stop and destroy the old camera properly
                    if camera.is_listening:
                        camera.stop()
                    camera.destroy()

                    # Create and attach a new camera to the selected vehicle
                    camera, camera_bp = attach_camera(world, new_vehicle, camera_transform)
                    camera.listen(lambda image: pygame_callback(image, render_object))

    if camera.is_listening:
        camera.stop()
    camera.destroy()
    pygame.quit()

def main():
    """
    Main function to initialize the CARLA simulation and run the game loop.
    """
    parser = argparse.ArgumentParser(description="CARLA Simulation with Ego Vehicle Visualization")
    parser.add_argument("--vis_ego", action="store_true", help="Visualize ego vehicle sensor data in a separate window")
    args = parser.parse_args()

    pygame.init()

    config = load_config("utils/config/config.yaml")
    logging.basicConfig(
        level=config.logging.level.upper(),
        filename=config.logging.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    client, world = initialize_carla()
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager, config)

    env_manager = EnvironmentManager(world)
    env_manager.cleanup_existing_actors()

    spawn_points = env_manager.filter_spawn_points(config.simulation.min_spawn_distance)
    vehicles = env_manager.spawn_with_retries(client, traffic_manager, spawn_points, config.simulation.num_vehicles, config.simulation.spawn_retries)

    if not vehicles:
        logging.error("No vehicles were spawned. Exiting simulation.")
        return

    ego_vehicle, smart_vehicles, vehicle_mapping = env_manager.designate_ego_and_smart_vehicles(vehicles, world, config)

    sensors_class = Sensors()
    sensor_suite = sensors_class.sensor_suite()
    ego_vehicle_sensors = attach_sensor_suite(world, ego_vehicle, sensor_suite)
    vehicle_mapping["ego_veh"]["sensors"] = ego_vehicle_sensors

    if args.vis_ego:
        visualize_ego_sensors(world, ego_vehicle_sensors)
        return

    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera, camera_bp = attach_camera(world, ego_vehicle, camera_transform)

    image_width = camera_bp.get_attribute("image_size_x").as_int()
    image_height = camera_bp.get_attribute("image_size_y").as_int()
    intrinsic_matrix = env_manager.get_camera_intrinsic(camera_bp, image_width, image_height)

    game_display = pygame.display.set_mode((image_width, image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    render_object = RenderObject(image_width, image_height)
    control_object = ControlObject(ego_vehicle)

    camera.listen(lambda image: pygame_callback(image, render_object))

    try:
        game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform, vehicle_mapping, intrinsic_matrix, env_manager)
    finally:
        pygame.quit()
        cleanup(client, vehicles, [])
        logging.info("Simulation ended. Cleaned up all resources.")


if __name__ == "__main__":
    main()