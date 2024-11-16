import carla
import logging

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
            if not vehicles:
                logging.warning("Spawn failed. Retrying...")
            attempt += 1

        if not vehicles:
            logging.error("Failed to spawn any vehicles after retries.")
        return vehicles
