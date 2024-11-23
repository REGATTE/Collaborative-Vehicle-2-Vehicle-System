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

    def filter_spawn_points(self, min_distance=20.0, visualize=True, life_time=10.0):
        """
        Filters spawn points to ensure minimum distance between vehicles.
        :param min_distance: Minimum distance between spawn points.
        :param visualize: Whether to visualize spawn points in the CARLA world.
        :param life_time: Lifetime for the visualized spawn points.
        """
        try:
            spawn_points = self.world.get_map().get_spawn_points()
            filtered_spawn_points = []
            for sp in spawn_points:
                if all(sp.location.distance(other.location) > min_distance for other in filtered_spawn_points):
                    filtered_spawn_points.append(sp)

            if visualize:
                for idx, sp in enumerate(filtered_spawn_points):
                    self.world.debug.draw_string(
                        sp.location, f"{idx}",
                        draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=life_time
                    )
            return filtered_spawn_points
        except Exception as e:
            logging.error(f"Error filtering spawn points: {e}", exc_info=True)
            return []

    def spawn_with_retries(self, client, traffic_manager, num_vehicles, spawn_retries):
        """
        Attempts to spawn the specified number of vehicles with retries.
        Captures spawn locations for use in vehicle mapping.
        :return: List of successfully spawned vehicles and their spawn locations.
        """
        logging.info(f"Starting to spawn {num_vehicles} vehicles with up to {spawn_retries} retries.")

        spawn_points = self.filter_spawn_points()
        if not spawn_points:
            logging.error("No spawn points available. Cannot spawn vehicles.")
            return []

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if not blueprints:
            logging.error("No vehicle blueprints available. Cannot spawn vehicles.")
            return []

        spawned_vehicles = []
        spawn_locations = []  # Added to track spawn locations
        for attempt in range(spawn_retries):
            logging.info(f"Spawn attempt {attempt + 1}/{spawn_retries}...")
            for sp in spawn_points:
                if len(spawned_vehicles) >= num_vehicles:
                    break

                try:
                    vehicle_bp = random.choice(blueprints)
                    vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                    if vehicle:
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        spawned_vehicles.append(vehicle)
                        spawn_locations.append(sp.location)  # Capture the spawn location
                        logging.info(f"Successfully spawned vehicle (ID: {vehicle.id}) at location {sp.location}.")
                    else:
                        logging.warning(f"Failed to spawn vehicle at {sp.location}.")
                except Exception as e:
                    logging.error(f"Error during vehicle spawn at {sp.location}: {e}")

            if len(spawned_vehicles) >= num_vehicles:
                logging.info("All vehicles successfully spawned.")
                break

        if len(spawned_vehicles) < num_vehicles:
            logging.warning(f"Only {len(spawned_vehicles)} out of {num_vehicles} vehicles were spawned.")

        return spawned_vehicles, spawn_locations

    def cleanup_existing_actors(self, vehicle_mapping=None):
        """
        Destroys all existing vehicles, pedestrians, and their attached sensors in the world.
        Optionally cleans up sensors associated with the provided vehicle mapping.
        :param vehicle_mapping: Optional dictionary of vehicle mappings to destroy associated sensors.
        """
        actors = self.world.get_actors()
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        
        # Destroy sensors associated with the vehicle mapping
        if vehicle_mapping:
            for label, data in vehicle_mapping.items():
                for sensor_id in data.get("sensors", []):
                    sensor = self.world.get_actor(sensor_id)
                    if sensor:
                        try:
                            sensor.destroy()
                            logging.info(f"Destroyed sensor ID: {sensor_id} associated with {label}.")
                        except Exception as e:
                            logging.error(f"Error destroying sensor ID {sensor_id}: {e}")

        # Destroy vehicles
        for vehicle in vehicles:
            try:
                vehicle.destroy()
                logging.info(f"Destroyed vehicle ID: {vehicle.id}")
            except Exception as e:
                logging.error(f"Error destroying vehicle ID {vehicle.id}: {e}")

        # Destroy pedestrians
        for pedestrian in pedestrians:
            try:
                pedestrian.destroy()
                logging.info(f"Destroyed pedestrian ID: {pedestrian.id}")
            except Exception as e:
                logging.error(f"Error destroying pedestrian ID {pedestrian.id}: {e}")

        logging.info(f"Cleaned up {len(vehicles)} vehicles and {len(pedestrians)} pedestrians.")

    def designate_ego_and_smart_vehicles(self, vehicles, spawn_locations, world, config):
        """
        Designates one vehicle as the ego vehicle and the rest as smart vehicles.
        Includes sensors cleanup if a previous simulation was run.
        Captures initial global positions and includes them in the vehicle mapping.
        :param vehicles: List of vehicle IDs spawned in the simulation.
        :param world: CARLA world object.
        :param config: Configuration object for simulation.
        :return: (ego_vehicle, smart_vehicles, vehicle_mapping)
        """
        vehicle_mapping = {}

        # Extract actor IDs if the vehicles list contains Vehicle objects
        vehicle_ids = [vehicle.id if hasattr(vehicle, 'id') else vehicle for vehicle in vehicles]

        # Select first vehicle as the ego vehicle
        ego_vehicle_id = vehicle_ids[0]  # First vehicle is designated as ego
        ego_vehicle = world.get_actor(ego_vehicle_id)
        ego_spawn_location = spawn_locations[0]
        vehicle_mapping["ego_veh"] = {
            "actor_id": ego_vehicle.id,
            "sensors": [],
            "initial_position": {
                "x": ego_spawn_location.x,
                "y": ego_spawn_location.y,
                "z": ego_spawn_location.z
            }
        }
        logging.info(f"Designated vehicle {ego_vehicle_id} as ego_veh with initial spawn position: {ego_spawn_location}.")

        # Assign remaining vehicles as smart vehicles
        smart_vehicle_ids = [vid for vid in vehicle_ids if vid != ego_vehicle_id]
        smart_vehicles = []
        for idx, (smart_vehicle_id, spawn_location) in enumerate(zip(smart_vehicle_ids, spawn_locations[1:]), start=1):
            # Validate if the smart vehicle exists
            smart_vehicle = world.get_actor(smart_vehicle_id)
            if smart_vehicle is None:
                logging.warning(f"Smart vehicle with ID {smart_vehicle_id} could not be found. Skipping.")
                continue

            vehicle_label = f"smart_veh_{idx}"
            vehicle_mapping[vehicle_label] = {
                "actor_id": smart_vehicle.id,
                "sensors": [],
                "initial_position": {
                    "x": spawn_location.x,
                    "y": spawn_location.y,
                    "z": spawn_location.z
                }
            }
            smart_vehicles.append(smart_vehicle)
            logging.info(f"Designated vehicle {smart_vehicle_id} as {vehicle_label} with initial spawn position: {spawn_location}.")

            # Limit the number of smart vehicles to the desired count
            if idx == config.simulation.num_smart_vehicles:
                break

        return ego_vehicle, smart_vehicles, vehicle_mapping

    def get_ego_vehicle(self, vehicle_mapping):
        """
        Retrieves the ego vehicle from the vehicle mapping.
        :param vehicle_mapping: A dictionary containing all vehicles and their labels.
        :return: The ego vehicle actor if found, None otherwise.
        """
        ego_vehicle_label = "ego_veh"
        if ego_vehicle_label in vehicle_mapping:
            return vehicle_mapping[ego_vehicle_label]["actor"]
        else:
            logging.error("Ego vehicle not found in the vehicle mapping.")
            return None

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

            # Use "actor_id" instead of "actor"
            vehicle_id_text = f"{label} (ID: {vehicle_data['actor_id']})"
            text_surface = font.render(vehicle_id_text, True, color)
            screen.blit(text_surface, (x_offset, 5))  # Render text at the top
            x_offset += spacing  # Increment x position for next label