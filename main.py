import carla
import random
import pygame
import numpy as np
import logging
import argparse

from agents.controller import ControlObject
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, cleanup
from Simulation.sensors import Sensors
from agents.EnvironmentManager import EnvironmentManager
from agents.visualize_ego import visualize_ego_sensors
from utils.config.config_loader import load_config
from utils.logging_config import configure_logging


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
        """
        Initializes the RenderObject with a surface and dimensions.
        :param width: The width of the PyGame surface.
        :param height: The height of the PyGame surface.
        """
        self.width = width
        self.height = height
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


def initialize_pygame(camera_bp, num_labels, fixed_resolution=(1920, 1080)):
    """
    Initializes the PyGame window with a fixed resolution or dynamic calculation.
    :param camera_bp: Blueprint of the camera.
    :param num_labels: Number of labels to display in the menu bar.
    :param fixed_resolution: Fixed resolution for the PyGame window.
    :return: PyGame display surface and adjusted resolution (width, height).
    """
    pygame.init()

    # Use the fixed resolution directly
    window_width, window_height = fixed_resolution
    game_display = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    game_display.fill((0, 0, 0))
    pygame.display.flip()

    return game_display, window_width, window_height

def pygame_callback(image, render_object):
    """
    Callback function to process camera images and render them in PyGame.
    :param image: Camera image from CARLA.
    :param render_object: RenderObject to display the image.
    """
    try:
        # Convert raw image data to RGB format
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha channel
        array = array[:, :, ::-1]  # Convert BGRA to RGB

        # Scale the image to fit the render object's surface dimensions
        scaled_image = pygame.transform.scale(
            pygame.surfarray.make_surface(array.swapaxes(0, 1)),
            (render_object.width, render_object.height)
        )

        # Update the render surface
        render_object.surface = scaled_image
    except Exception as e:
        logging.error(f"Error in pygame_callback: {e}")


def attach_camera(world, vehicle, camera_transform):
    """
    Attaches a camera sensor to a vehicle with a specified transform.
    :param world: CARLA world object.
    :param vehicle: CARLA vehicle actor to attach the camera to.
    :param camera_transform: Transform object specifying the camera's position and orientation.
    :return: The attached camera actor.
    """
    try:
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')

        # Spawn the camera sensor
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logging.info(f"Car following camera attached to Vehicle ID: {vehicle.id}")
        return camera
    except Exception as e:
        logging.error(f"Error attaching camera to Vehicle ID: {vehicle.id}: {e}")
        return None

def game_loop(world, game_display, camera, render_object, control_object, vehicle_mapping, env_manager):
    """
    Main game loop for updating the CARLA world and PyGame display.

    This loop handles:
        - Rendering the PyGame window with the camera feed.
        - Drawing the menu bar with vehicle IDs.
        - Switching vehicles using the TAB key.
        - Processing user input for vehicle control.

    :param world: CARLA world object.
    :param game_display: PyGame display surface.
    :param camera: Camera actor attached to a vehicle.
    :param render_object: RenderObject for rendering the camera feed.
    :param control_object: ControlObject for handling vehicle control inputs.
    :param vehicle_mapping: Dictionary mapping vehicle labels to their actors and sensors.
    :param env_manager: EnvironmentManager for drawing vehicle labels.
    """
    crashed = False
    font = pygame.font.SysFont('Arial', 16)
    menu_bar_height = 30
    width, height = game_display.get_size()

    # List of vehicle labels for toggling
    vehicle_keys = list(vehicle_mapping.keys())
    current_vehicle_index = 0
    active_vehicle_label = vehicle_keys[current_vehicle_index]

    logging.info("Entering the main simulation loop.")
    while not crashed:
        world.tick()  # Advance simulation by one tick
        game_display.fill((0, 0, 0))  # Clear the screen

        # Render the camera feed
        if render_object.surface:
            game_display.blit(render_object.surface, (0, menu_bar_height))  # Offset for menu bar height

        # Render the menu bar
        env_manager.draw_vehicle_labels_menu_bar(game_display, font, vehicle_mapping, width, active_vehicle_label)

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
                    active_vehicle_label = vehicle_keys[current_vehicle_index]
                    logging.info(f"Switched to vehicle: {active_vehicle_label}")

                    # Reattach the camera to the newly selected vehicle
                    new_vehicle = vehicle_mapping[active_vehicle_label]["actor"]
                    if camera.is_listening:
                        camera.stop()
                    camera.destroy()
                    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
                    camera = attach_camera(world, new_vehicle, camera_transform)
                    camera.listen(lambda image: pygame_callback(image, render_object))

    if camera.is_listening:
        camera.stop()
    camera.destroy()
    pygame.quit()

def main():
    """
    Entry point for the CARLA simulation.

    Features:
        - Initializes CARLA client and server connection.
        - Configures synchronous mode for deterministic simulation.
        - Spawns vehicles and designates an ego vehicle and smart vehicles.
        - Attaches sensors to all vehicles and logs their IDs.
        - Provides interactive visualization and control via PyGame.
    """
    configure_logging()
    logging.info("Starting CARLA simulation...")

    parser = argparse.ArgumentParser(description="CARLA Simulation with Sensor Data Association")
    parser.add_argument("--vis_ego", action="store_true", help="Visualize ego vehicle sensors")
    parser.add_argument("--vis_smart", action="store_true", help="Visualize smart vehicle sensors with toggling")
    args = parser.parse_args()

    # Load configuration file
    config = load_config("utils/config/config.yaml")

    # Initialize CARLA client and world
    client, world = initialize_carla()
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager, config)

    # Clean up any pre-existing actors
    env_manager = EnvironmentManager(world)
    env_manager.cleanup_existing_actors()

    # Spawn vehicles and assign IDs
    logging.info(f"Attempting to spawn {config.simulation.num_vehicles} vehicles.")
    vehicles = env_manager.spawn_with_retries(
        client, traffic_manager, config.simulation.num_vehicles, config.simulation.spawn_retries
    )
    if len(vehicles) < config.simulation.num_vehicles:
        logging.warning("Not all vehicles could be spawned. Check spawn points or increase retries.")
    logging.info(f"Successfully spawned {len(vehicles)} vehicles.")

    # Designate ego and smart vehicles
    ego_vehicle, smart_vehicles, vehicle_mapping = env_manager.designate_ego_and_smart_vehicles(vehicles, world, config)

    # Attach sensors to all vehicles
    sensors = Sensors()
    ego_vehicle_sensors = sensors.attach_sensor_suite(world, ego_vehicle, "ego_veh")
    logging.info(f"Ego vehicle (ID: {ego_vehicle.id}) has {len(ego_vehicle_sensors)} sensors attached.")

    for idx, smart_vehicle in enumerate(smart_vehicles, start=1):
        vehicle_label = f"smart_veh_{idx}"
        smart_sensors = sensors.attach_sensor_suite(world, smart_vehicle, vehicle_label)
        vehicle_mapping[vehicle_label]["sensors"] = smart_sensors
        logging.info(f"{vehicle_label} (ID: {smart_vehicle.id}) has {len(smart_sensors)} sensors attached.")
    vehicle_mapping["ego_veh"]["sensors"] = ego_vehicle_sensors

    # Visualization options
    if args.vis_ego:
        visualize_ego_sensors(world, ego_vehicle_sensors)
        return

    # Attach a camera to the ego vehicle for interactive visualization
    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera = attach_camera(world, ego_vehicle, camera_transform)
    game_display, width, height = initialize_pygame(None, len(vehicle_mapping), fixed_resolution=(1920, 1080))
    render_object = RenderObject(width, height - 30)
    control_object = ControlObject(ego_vehicle)

    # Start listening to the camera
    camera.listen(lambda image: pygame_callback(image, render_object))
    try:
        game_loop(world, game_display, camera, render_object, control_object, vehicle_mapping, env_manager)
    finally:
        pygame.quit()  # Close the PyGame window
        cleanup(client, vehicles, [])  # Clean up actors

if __name__ == "__main__":
    main()
