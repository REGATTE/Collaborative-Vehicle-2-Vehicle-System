import carla
import random
import pygame
import numpy as np
import logging
import argparse
import threading

from agents.controller import ControlObject
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, cleanup
from Simulation.sensors import Sensors  # Import Sensors class for the sensor suite
from agents.EnvironmentManager import EnvironmentManager
from agents.visualize_ego import visualize_ego_sensors

from utils.config.config_loader import load_config

# Function to initialize CARLA client and world
def initialize_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


# Function to set up CARLA in synchronous mode
def setup_synchronous_mode(world, traffic_manager, config):
    settings = world.get_settings()
    settings.synchronous_mode = config.simulation.synchronous_mode #synchronous mode check
    settings.fixed_delta_seconds = config.simulation.fixed_delta_seconds #tick time
    world.apply_settings(settings)

    traffic_manager.set_global_distance_to_leading_vehicle(5.0)  # Maintain a safe distance
    traffic_manager.set_synchronous_mode(config.simulation.synchronous_mode)  # Keep consistent simulation timing
    traffic_manager.set_random_device_seed(0)  # Consistent behavior
    random.seed(0)


# Class for rendering images to PyGame
class RenderObject:
    """
    A class to handle rendering images from the camera to a PyGame surface.
    """
    def __init__(self, width, height):
        # Initialize with a random image to prevent crashes
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


# Function to attach a suite of sensors to a vehicle
def attach_sensor_suite(world, vehicle, sensor_suite):
    """
    Attaches the provided sensor suite to the vehicle.
    """
    sensors = []
    for sensor_spec in sensor_suite:
        # Get the sensor blueprint
        sensor_bp = world.get_blueprint_library().find(sensor_spec['type'])

        # Set up the transform with default or provided values
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

        # Spawn the sensor and attach it to the vehicle
        sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)

        # Apply additional attributes if defined
        for key, value in sensor_spec.items():
            if sensor_bp.has_attribute(key):
                sensor_bp.set_attribute(key, str(value))

        # Add sensor to the list and log it
        sensors.append(sensor)
        logging.info(f"Attached sensor {sensor_spec['type']} to vehicle {vehicle.id}.")

    return sensors


# Function to spawn and attach sensors
def spawn_and_attach_sensors(client, world, traffic_manager, spawn_points, number_of_vehicles):
    vehicles = spawn_vehicles(client, world, traffic_manager, number_of_vehicles=number_of_vehicles)
    if not vehicles:
        logging.error("Failed to spawn any vehicles.")
        return [], {}

    sensors_class = Sensors()
    sensor_suite = sensors_class.sensor_suite()

    sensor_attachments = {}
    for vehicle_id in vehicles:
        vehicle = world.get_actor(vehicle_id)
        sensors = attach_sensor_suite(world, vehicle, sensor_suite)
        sensor_attachments[vehicle_id] = sensors

    return vehicles, sensor_attachments


def designate_ego_and_smart_vehicles(vehicles, world, config):
    """
    Designates one vehicle as the ego vehicle and the rest as smart vehicles.
    Adds labels to each vehicle in the simulation with increased height for better visibility.
    Maintains a mapping of vehicle IDs to their actors and sensors.
    :param vehicles: List of spawned vehicle actor IDs.
    :param world: CARLA world instance.
    :param config: Configuration settings.
    :return: Tuple of ego vehicle actor, list of smart vehicle actors, and a vehicle mapping.
    """
    vehicle_mapping = {}  # Dictionary to store mapping of vehicle ID to actor and sensors

    # Select and label the ego vehicle
    ego_vehicle_id = random.choice(vehicles)
    ego_vehicle = world.get_actor(ego_vehicle_id)
    logging.info(f"Designated vehicle {ego_vehicle.id} as the main ego vehicle.")

    ego_location = ego_vehicle.get_location()
    ego_location.z += 3.0  # Raise the label above the vehicle
    world.debug.draw_string(
        ego_location,
        "EGO_VEHICLE",
        draw_shadow=False,
        color=carla.Color(*config.colors.ego_vehicle),
        life_time=0,  # Persistent label
        persistent_lines=config.simulation.debug_labels
    )
    vehicle_mapping["EGO"] = {"actor": ego_vehicle, "sensors": []}  # Initialize sensors list for EGO

    # Select and label the smart vehicles
    smart_vehicle_ids = [vid for vid in vehicles if vid != ego_vehicle_id]
    smart_vehicles = [world.get_actor(vid) for vid in smart_vehicle_ids]

    for idx, smart_vehicle in enumerate(smart_vehicles, start=1):
        smart_location = smart_vehicle.get_location()
        smart_location.z += 3.0  # Raise the label above the vehicle
        label = f"SMART_CAR_{idx}"
        world.debug.draw_string(
            smart_location,
            label,
            draw_shadow=False,
            color=carla.Color(*config.colors.smart_vehicle),
            life_time=0,  # Persistent label
            persistent_lines=config.simulation.debug_labels
        )
        logging.info(f"Designated vehicle {smart_vehicle.id} as {label}.")
        vehicle_mapping[label] = {"actor": smart_vehicle, "sensors": []}  # Initialize sensors list for smart car

    return ego_vehicle, smart_vehicles, vehicle_mapping

# Function to initialize Pygame
def initialize_pygame(camera_bp):
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    pygame.init()
    game_display = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    game_display.fill((0, 0, 0))
    pygame.display.flip()

    return game_display, image_w, image_h


# Function to process camera feed for Pygame
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]  # Extract RGB channels
    img = img[:, :, ::-1]  # Convert from BGRA to RGB
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


# Function to attach a camera to the ego vehicle
def attach_camera(world, ego_vehicle, camera_transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    return camera, camera_bp


# Function for the game loop
def game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform):
    crashed = False
    while not crashed:
        world.tick()

        # Update the display with the camera feed
        game_display.blit(render_object.surface, (0, 0))
        pygame.display.flip()

        # Process control state
        control_object.process_control()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            elif event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                # Switch to another random vehicle
                current_vehicle = control_object.vehicle
                current_vehicle.set_autopilot(True)  # Enable autopilot for the current vehicle

                # Choose a new vehicle and retrieve its Actor object
                new_vehicle_id = random.choice(vehicles)
                new_vehicle = world.get_actor(new_vehicle_id)
                if new_vehicle and new_vehicle.is_alive:
                    # Stop and destroy the previous camera
                    camera.stop()
                    camera.destroy()

                    # Update the control object and attach a new camera
                    control_object = ControlObject(new_vehicle)
                    camera, _ = attach_camera(world, new_vehicle, camera_transform)
                    camera.listen(lambda image: pygame_callback(image, render_object))

                    # Reset the display
                    game_display.fill((0, 0, 0))
                    pygame.display.flip()

    # Cleanup after exiting the loop
    camera.stop()
    camera.destroy()
    pygame.quit()


# Main function
def main():
    parser = argparse.ArgumentParser(description="CARLA Simulation with Ego Vehicle Visualization")
    parser.add_argument("--vis_ego", action="store_true", help="Visualize ego vehicle sensor data in a separate window")
    args = parser.parse_args()

    # Load configuration
    config = load_config("utils/config/config.yaml")
    logging.basicConfig(
        level=config.logging.level.upper(),
        filename=config.logging.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize CARLA
    client, world = initialize_carla()

    # Setup synchronous mode
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager, config)

    # Cleanup existing actors
    env_manager = EnvironmentManager(world)
    env_manager.cleanup_existing_actors()

    # Spawn vehicles
    spawn_points = env_manager.filter_spawn_points(config.simulation.min_spawn_distance)
    vehicles = env_manager.spawn_with_retries(client, traffic_manager, spawn_points, config.simulation.num_vehicles, config.simulation.spawn_retries)
    if not vehicles:
        logging.error("No vehicles were spawned. Exiting simulation.")
        return

    # Designate ego and smart vehicles
    ego_vehicle, smart_vehicles, vehicle_mapping = designate_ego_and_smart_vehicles(vehicles, world, config)

    # Attach sensors to ego vehicle
    sensors_class = Sensors()
    sensor_suite = sensors_class.sensor_suite()
    ego_vehicle_sensors = attach_sensor_suite(world, ego_vehicle, sensor_suite)
    vehicle_mapping["EGO"]["sensors"] = ego_vehicle_sensors

    if args.vis_ego:
        logging.info("Launching ego vehicle sensor visualization.")
        visualize_ego_sensors(world, ego_vehicle_sensors)
        return  # Exit after visualization loop

    # Continue with the original simulation loop
    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera, camera_bp = attach_camera(world, ego_vehicle, camera_transform)
    game_display, image_w, image_h = initialize_pygame(camera_bp)
    render_object = RenderObject(image_w, image_h)
    control_object = ControlObject(ego_vehicle)

    camera.listen(lambda image: pygame_callback(image, render_object))

    try:
        game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform)
    finally:
        cleanup(client, vehicles, [])
        logging.info("Simulation ended. Cleaned up all resources.")

if __name__ == "__main__":
    main()
