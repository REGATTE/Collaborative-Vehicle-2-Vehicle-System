import carla
import logging
import math
import json
import socket


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

def find_vehicles_in_radius(ego_vehicle, smart_vehicles, radius=20.0):
    """
    Find all smart vehicles within a given radius of the ego vehicle.
    :param ego_vehicle: Ego vehicle actor.
    :param smart_vehicles: List of smart vehicle actors.
    :param radius: Radius in meters to consider for proximity.
    :return: List of tuples containing smart vehicles and their distances.
    """
    ego_location = ego_vehicle.get_transform().location
    vehicles_in_radius = []

    for smart_vehicle in smart_vehicles:
        smart_location = smart_vehicle.get_transform().location
        distance = calculate_distance(ego_location, smart_location)
        if distance < radius:
            vehicles_in_radius.append((smart_vehicle.id, smart_vehicle, distance))

    return vehicles_in_radius

def send_data_to_ego(ego_address, smart_vehicle_id, smart_vehicle):
    """
    Sends data from the smart vehicle to the ego vehicle.
    :param ego_address: Tuple of (IP, Port) for the ego vehicle.
    :param smart_vehicle_id: ID of the smart vehicle.
    :param smart_vehicle: Smart vehicle actor.
    """
    location = smart_vehicle.get_transform().location
    position = {
        "x": location.x,
        "y": location.y,
        "z": location.z
    }
    smart_data = {
        "id": smart_vehicle_id,
        "position": position,
        "speed": smart_vehicle.get_velocity().length()
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(ego_address)
            sock.sendall(json.dumps(smart_data).encode())
    except Exception as e:
        logging.error(f"Error sending data from Smart Vehicle {smart_vehicle_id} to Ego Vehicle: {e}")

def log_proximity_and_trigger_communication(ego_vehicle, smart_vehicles, world, proximity_state):
    """
    Logs the IDs of all smart vehicles in proximity and triggers one-way communication.
    :param ego_vehicle: Ego vehicle actor.
    :param smart_vehicles: List of smart vehicle actors.
    :param world: CARLA world object.
    :param proximity_state: Dictionary to track logged vehicles.
    """
    vehicles_in_radius = find_vehicles_in_radius(ego_vehicle, smart_vehicles)
    ego_address = ('127.0.0.1', 65432)  # Ego Vehicle's listening address

    for smart_vehicle_id, smart_vehicle, distance in vehicles_in_radius:
        if smart_vehicle_id not in proximity_state:
            logging.info(
                f"Smart Vehicle ID {smart_vehicle_id} is close to Ego Vehicle (ID: {ego_vehicle.id}) "
                f"at {distance:.2f}m."
            )
            proximity_state[smart_vehicle_id] = True
            # Smart vehicle sends its data to the ego vehicle
            send_data_to_ego(ego_address, smart_vehicle_id, smart_vehicle)

    # Remove vehicles that have moved out of radius
    proximity_state = {
        vid: state for vid, state in proximity_state.items()
        if vid in [v_id for v_id, _, _ in vehicles_in_radius]
    }