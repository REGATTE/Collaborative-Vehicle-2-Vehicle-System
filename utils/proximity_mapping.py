import carla
import logging
import math


def calculate_distance(location1, location2):
    """
    Calculate the Euclidean distance between two CARLA locations.
    :param location1: The first location.
    :param location2: The second location.
    :return: Distance in meters.
    """
    return math.sqrt(
        (location1.x - location2.x) ** 2 +
        (location1.y - location2.y) ** 2 +
        (location1.z - location2.z) ** 2
    )


def find_closest_smart_vehicle(world, ego_vehicle, smart_vehicles, radius=20.0):
    """
    Find the closest smart vehicle to the ego vehicle within a given radius.
    :param world: CARLA world object.
    :param ego_vehicle: Ego vehicle actor.
    :param smart_vehicles: List of smart vehicle actors.
    :param radius: Radius in meters to consider for proximity.
    :return: The closest smart vehicle actor within the radius, or None.
    """
    ego_location = ego_vehicle.get_transform().location
    closest_vehicle = None
    min_distance = float('inf')

    for smart_vehicle in smart_vehicles:
        smart_location = smart_vehicle.get_transform().location
        distance = calculate_distance(ego_location, smart_location)
        if distance < radius and distance < min_distance:
            closest_vehicle = smart_vehicle
            min_distance = distance

    return closest_vehicle, min_distance


def log_proximity_mapping(ego_vehicle, smart_vehicles, world, proximity_state):
    """
    Logs the ID of the closest smart vehicle to the ego vehicle if within the radius.
    Only logs when a vehicle enters the proximity radius.
    :param ego_vehicle: Ego vehicle actor.
    :param smart_vehicles: List of smart vehicle actors.
    :param world: CARLA world object.
    :param proximity_state: Dictionary to track which vehicles are already logged.
    """
    closest_vehicle, distance = find_closest_smart_vehicle(world, ego_vehicle, smart_vehicles)
    
    if closest_vehicle:
        if closest_vehicle.id not in proximity_state:
            logging.info(f"Smart Vehicle ID {closest_vehicle.id} is closest to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m.")
            proximity_state[closest_vehicle.id] = True
    else:
        # Reset the state if no vehicles are within proximity
        proximity_state.clear()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Proximity Mapping...")

    # Assume `world`, `ego_vehicle`, and `smart_vehicles` are initialized through the main environment setup.
    # Replace the following placeholders with actual references if running standalone.
    world = None  # Replace with initialized CARLA world
    ego_vehicle = None  # Replace with ego vehicle actor
    smart_vehicles = []  # Replace with list of smart vehicle actors

    if world and ego_vehicle and smart_vehicles:
        log_proximity_mapping(ego_vehicle, smart_vehicles, world)
    else:
        logging.error("World, Ego Vehicle, or Smart Vehicles not initialized.")
