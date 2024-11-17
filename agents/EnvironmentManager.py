import carla
import logging
import random
import numpy as np
import pygame

from Simulation.generate_traffic import spawn_vehicles

class EnvironmentManager:
    """
    A class to manage the CARLA environment, including cleaning up existing actors,
    filtering valid spawn points, and retrying vehicle spawns.
    """

    def __init__(self, world):
        """
        Initialize the manager with the CARLA world.
        :param world: The CARLA world instance.
        """
        self.world = world

    def cleanup_existing_actors(self):
        """
        Destroys all existing vehicles and pedestrians in the world.
        """
        actors = self.world.get_actors()
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')

        for vehicle in vehicles:
            vehicle.destroy()
        for pedestrian in pedestrians:
            pedestrian.destroy()

        logging.info(f"Cleaned up {len(vehicles)} vehicles and {len(pedestrians)} pedestrians.")

    def filter_spawn_points(self, min_distance=10.0):
        """
        Filters spawn points to ensure a minimum distance between them.
        :param min_distance: Minimum distance between spawn points in meters.
        :return: List of filtered spawn points.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        filtered_spawn_points = []
        for sp in spawn_points:
            if all(sp.location.distance(other.location) > min_distance for other in filtered_spawn_points):
                filtered_spawn_points.append(sp)

        # Visualize spawn points
        for idx, sp in enumerate(filtered_spawn_points):
            self.world.debug.draw_string(sp.location, f"{idx}", draw_shadow=False,
                                         color=carla.Color(r=0, g=255, b=0), life_time=10.0)
        logging.info(f"Filtered {len(filtered_spawn_points)} valid spawn points.")
        return filtered_spawn_points

    def spawn_with_retries(self, client, traffic_manager, spawn_points, number_of_vehicles, retries=3):
        """
        Attempt to spawn vehicles with retries in case of failure.
        :param client: The CARLA client instance.
        :param traffic_manager: The CARLA traffic manager instance.
        :param spawn_points: List of valid spawn points.
        :param number_of_vehicles: Number of vehicles to spawn.
        :param retries: Number of retry attempts in case of failure.
        :return: List of spawned vehicle IDs.
        """
        attempt = 0
        vehicles = []

        while attempt < retries and not vehicles:
            logging.info(f"Spawn attempt {attempt + 1}/{retries}...")
            vehicles = spawn_vehicles(client, self.world, traffic_manager, number_of_vehicles=number_of_vehicles)

            if vehicles:
                logging.info(f"Successfully spawned {len(vehicles)} vehicles.")
            else:
                logging.warning("Spawn failed. Retrying...")
            
            attempt += 1

        if not vehicles:
            logging.error("Failed to spawn any vehicles after retries.")

        return vehicles


    
    def designate_ego_and_smart_vehicles(self, vehicles, world, config):
        """
        Designates one vehicle as the ego vehicle and the rest as smart vehicles.
        Adds labels to each vehicle in the simulation with increased height for better visibility.
        Maintains a mapping of vehicle IDs to their actors and sensors.
        Prints the mapping of roles to vehicle IDs.
        """
        vehicle_mapping = {}

        # Select and label the ego vehicle
        ego_vehicle_id = random.choice(vehicles)
        ego_vehicle = world.get_actor(ego_vehicle_id)
        logging.info(f"Designated vehicle {ego_vehicle.id} as ego_veh.")

        # Add label for the ego vehicle
        ego_location = ego_vehicle.get_transform().location
        ego_location.z += 3.0  # Raise the label above the vehicle
        world.debug.draw_string(
            ego_location,
            "ego_veh",
            draw_shadow=False,
            color=carla.Color(*config.colors.ego_vehicle),
            life_time=0,  # Persistent label
            persistent_lines=config.simulation.debug_labels
        )
        vehicle_mapping["ego_veh"] = {"actor": ego_vehicle, "sensors": []}

        # Print the ID of the ego vehicle
        print(f"ego_veh: Vehicle ID {ego_vehicle.id}")

        # Select and label the smart vehicles
        smart_vehicle_ids = [vid for vid in vehicles if vid != ego_vehicle_id]
        smart_vehicles = [world.get_actor(vid) for vid in smart_vehicle_ids]

        for idx, smart_vehicle in enumerate(smart_vehicles, start=1):
            smart_location = smart_vehicle.get_transform().location
            smart_location.z += 3.0  # Raise the label above the vehicle
            label = f"smart_veh_{idx}"
            world.debug.draw_string(
                smart_location,
                label,
                draw_shadow=False,
                color=carla.Color(*config.colors.smart_vehicle),
                life_time=0,  # Persistent label
                persistent_lines=config.simulation.debug_labels
            )
            logging.info(f"Designated vehicle {smart_vehicle.id} as {label}.")
            vehicle_mapping[label] = {"actor": smart_vehicle, "sensors": []}

            # Print the ID of each smart vehicle
            print(f"{label}: Vehicle ID {smart_vehicle.id}")

        return ego_vehicle, smart_vehicles, vehicle_mapping


    def get_camera_intrinsic(self, camera_bp, image_width, image_height):
        """
        Calculates the camera intrinsic matrix based on the camera blueprint and resolution.
        :param camera_bp: Camera blueprint.
        :param image_width: Width of the camera's output.
        :param image_height: Height of the camera's output.
        :return: 3x3 intrinsic matrix.
        """
        fov = float(camera_bp.get_attribute("fov").as_float())
        focal_length = image_width / (2.0 * np.tan(np.radians(fov) / 2.0))

        intrinsic_matrix = np.array([
            [focal_length, 0, image_width / 2.0],
            [0, focal_length, image_height / 2.0],
            [0, 0, 1]
        ])
        return intrinsic_matrix

    def draw_vehicle_labels_menu_bar(self, screen, font, vehicle_mapping, width, active_vehicle_label):
        """
        Displays labels for all vehicles (ego and smart) on a static menu bar at the top of the PyGame window.
        Highlights the currently active vehicle.
        :param screen: PyGame display surface.
        :param font: PyGame font for text rendering.
        :param vehicle_mapping: Dictionary of vehicles and their roles (e.g., ego_veh, smart_veh_1).
        :param width: Width of the PyGame window.
        :param active_vehicle_label: Label of the currently active vehicle.
        """
        # Create a menu bar area
        menu_height = 30  # Height of the menu bar
        pygame.draw.rect(screen, (50, 50, 50), (0, 0, width, menu_height))  # Dark gray menu bar

        # Calculate dynamic spacing
        num_labels = len(vehicle_mapping)
        spacing = max(10, (width - 20) // num_labels)  # Ensure a minimum spacing of 10 pixels

        # Display labels in the menu bar
        x_offset = 10  # Starting x position for text
        for label, vehicle_data in vehicle_mapping.items():
            # Highlight the active vehicle
            color = (255, 255, 0) if label == active_vehicle_label else (255, 255, 255)  # Yellow for active, white otherwise
            vehicle_id_text = f"{label} (ID: {vehicle_data['actor'].id})"
            text_surface = font.render(vehicle_id_text, True, color)
            screen.blit(text_surface, (x_offset, 5))  # Render text at the top
            x_offset += spacing  # Increment x position for next label

    
    def draw_vehicle_labels(self, screen, font, camera, vehicle_mapping, intrinsic_matrix, width, height):
        """
        Draws labels for all vehicles (ego and smart) on the PyGame window.
        Displays their IDs above the vehicles in the simulation.
        """
        def world_to_screen(world_pos, intrinsic_matrix, camera_transform):
            """
            Converts a 3D world position to 2D screen coordinates using the camera's intrinsic matrix.
            """
            # Get the relative position of the vehicle to the camera
            camera_world_matrix = camera_transform.get_matrix()
            camera_relative_matrix = np.linalg.inv(camera_world_matrix)
            relative_pos = np.dot(camera_relative_matrix, np.array([world_pos.x, world_pos.y, world_pos.z, 1]))

            # Ignore objects behind the camera
            if relative_pos[2] <= 0:
                return None

            # Project 3D coordinates to 2D screen space
            screen_pos = np.dot(intrinsic_matrix, relative_pos[:3])
            screen_pos /= screen_pos[2]

            return int(screen_pos[0]), int(screen_pos[1])

        # Get the camera's transform
        camera_transform = camera.get_transform()

        for label, vehicle_data in vehicle_mapping.items():
            vehicle = vehicle_data["actor"]
            vehicle_pos = vehicle.get_transform().location
            vehicle_pos.z += 2.5  # Offset the label above the vehicle for visibility

            # Convert world coordinates to screen coordinates
            screen_pos = world_to_screen(vehicle_pos, intrinsic_matrix, camera_transform)
            if screen_pos:
                x, y = screen_pos

                # Ensure the screen position is within bounds
                if 0 <= x < width and 0 <= y < height:
                    # Prepare the label text (role and vehicle ID)
                    vehicle_id_text = f"{label} (ID: {vehicle.id})"

                    # Render the label on the screen
                    try:
                        text_surface = font.render(vehicle_id_text, True, (255, 255, 255))  # White text
                        screen.blit(text_surface, (x, y))
                    except Exception as e:
                        logging.error(f"Error rendering label for {label}: {e}")




