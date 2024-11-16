import carla
import random
import pygame
import numpy as np
import logging
from agents.controller import ControlObject
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, spawn_walkers, cleanup

# Function to initialize CARLA client and world
def initialize_carla():
    client = carla.Client('localhost', 2000)
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

# Function to attach a camera sensor to a vehicle
def attach_camera(world, ego_vehicle, camera_transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    return camera, camera_bp

# Function to initialize Pygame
def initialize_pygame(camera_bp):
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    pygame.init()
    game_display = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    game_display.fill((0, 0, 0))
    pygame.display.flip()

    return game_display, image_w, image_h

# Callback function to process camera images for Pygame
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]  # Extract RGB channels
    img = img[:, :, ::-1]  # Convert from BGRA to RGB
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

# Class to manage Pygame rendering
class RenderObject:
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

def game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform):
    """
    Main game loop for the simulation.
    """
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


def main():
    # Initialize CARLA
    client, world = initialize_carla()

    # Setup synchronous mode
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager)

    # Spawn vehicles and walkers
    vehicles = []
    walkers = []
    try:
        models = ['model3', 'mustang']
        logging.info("Spawning vehicles...")
        vehicles = spawn_vehicles(client, world, traffic_manager, number_of_vehicles=10)
        
        logging.info("Spawning walkers...")
        walkers, walker_speeds = spawn_walkers(client, world, number_of_walkers=5)

        # Select a random vehicle as the ego vehicle
        ego_vehicle = world.get_actor(random.choice(vehicles))

        # Attach camera to the ego vehicle
        camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
        camera, camera_bp = attach_camera(world, ego_vehicle, camera_transform)

        # Initialize Pygame
        game_display, image_w, image_h = initialize_pygame(camera_bp)

        # Create render and control objects
        render_object = RenderObject(image_w, image_h)
        control_object = ControlObject(ego_vehicle)
        print(f"ControlObject initialized with vehicle: {control_object.vehicle}")

        # Start the camera with the Pygame callback
        camera.listen(lambda image: pygame_callback(image, render_object))

        # Start the game loop
        game_loop(world, game_display, camera, render_object, control_object, vehicles, camera_transform)

    finally:
        # Ensure camera cleanup
        if camera is not None:
            camera.stop()
            camera.destroy()
        # Clean up all spawned actors
        cleanup(client, vehicles, walkers)
        logging.info("Cleaned up all actors.")
        pygame.quit()


if __name__ == "__main__":
    main()
