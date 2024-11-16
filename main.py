import carla
import random
import pygame
import numpy as np
import logging
from agents.controller import ControlObject
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, cleanup
from Simulation.sensors import Sensors  # Import Sensors class for the sensor suite

from agents.EnvironmentManager import EnvironmentManager


# Function to initialize CARLA client and world
def initialize_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


# Function to set up CARLA in synchronous mode
def setup_synchronous_mode(world, traffic_manager):
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # Simulation time step
    world.apply_settings(settings)

    traffic_manager.set_global_distance_to_leading_vehicle(5.0)  # Maintain a safe distance
    traffic_manager.set_synchronous_mode(True)  # Keep consistent simulation timing
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


def designate_ego_and_smart_vehicles(vehicles, world):
    """
    Designates one vehicle as the ego vehicle and the rest as smart vehicles.
    Adds labels to each vehicle in the simulation with increased height for better visibility.
    """
    ego_vehicle_id = random.choice(vehicles)
    ego_vehicle = world.get_actor(ego_vehicle_id)
    logging.info(f"Designated vehicle {ego_vehicle.id} as the main ego vehicle.")

    # Add a label for the ego vehicle
    ego_location = ego_vehicle.get_location()
    ego_location.z += 3.0  # Raise the label above the vehicle
    world.debug.draw_string(
        ego_location,
        "EGO_VEHICLE",
        draw_shadow=False,
        color=carla.Color(r=255, g=0, b=0),  # Red label for the ego vehicle
        life_time=0,  # Persistent label
        persistent_lines=True
    )

    smart_vehicle_ids = [vid for vid in vehicles if vid != ego_vehicle_id]
    smart_vehicles = [world.get_actor(vid) for vid in smart_vehicle_ids]

    # Add labels for smart vehicles
    for idx, smart_vehicle in enumerate(smart_vehicles, start=1):
        smart_location = smart_vehicle.get_location()
        smart_location.z += 3.0  # Raise the label above the vehicle
        world.debug.draw_string(
            smart_location,
            f"SMART_CAR_{idx}",
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0),  # Green labels for smart vehicles
            life_time=0,  # Persistent label
            persistent_lines=True
        )
        logging.info(f"Designated vehicle {smart_vehicle.id} as SMART_CAR_{idx}.")

    return ego_vehicle, smart_vehicles




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
    client, world = initialize_carla()
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager)

    # Initialize the EnvironmentManager
    env_manager = EnvironmentManager(world)

    # Cleanup existing actors
    env_manager.cleanup_existing_actors()

    # Filter spawn points
    spawn_points = env_manager.filter_spawn_points(min_distance=10.0)

    # Spawn vehicles with retries
    max_vehicles = min(5, len(spawn_points))
    vehicles = env_manager.spawn_with_retries(client, traffic_manager, spawn_points, number_of_vehicles=max_vehicles, retries=3)
    if not vehicles:
        logging.error("No vehicles were spawned. Exiting simulation.")
        return

    # Designate ego and smart vehicles
    ego_vehicle, smart_vehicles = designate_ego_and_smart_vehicles(vehicles, world)

    # Attach camera to ego vehicle
    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera, camera_bp = attach_camera(world, ego_vehicle, camera_transform)

    # Initialize Pygame
    game_display, image_w, image_h = initialize_pygame(camera_bp)

    # Create render and control objects
    render_object = RenderObject(image_w, image_h)
    control_object = ControlObject(ego_vehicle)
    logging.info(f"ControlObject initialized with vehicle: {control_object.vehicle}")

    # Start the camera
    camera.listen(lambda image: pygame_callback(image, render_object))

    try:
        # Start the game loop
        game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform)
    finally:
        cleanup(client, vehicles, [])
        logging.info("Simulation ended. Cleaned up all resources.")

if __name__ == "__main__":
    main()
