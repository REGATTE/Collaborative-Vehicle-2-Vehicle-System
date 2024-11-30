import logging
import carla
import socket
import json
from threading import Lock, Thread
from math import sqrt
import time
import os, cv2
import numpy as np

from utils.compression import DataCompressor

class ProximityMapping:
    def __init__(self, world, radius=20.0):
        """
        Initializes the proximity mapping class.
        :param world: CARLA world object.
        :param radius: Radius in meters to consider for proximity.
        """
        self.world = world
        self.radius = radius
        self.proximity_cache = {}
        logging.info(f"ProximityMapping initialized with radius: {self.radius} meters.")
        logging.info("=========================================")

    def calculate_distance(self, location1, location2):
        """
        Calculate the Euclidean distance between two CARLA locations.
        :param location1: The first location.
        :param location2: The second location.
        :return: Distance in meters.
        """
        return sqrt(
            (location1.x - location2.x) ** 2 +
            (location1.y - location2.y) ** 2 +
            (location1.z - location2.z) ** 2
        )

    def find_vehicles_in_radius(self, ego_vehicle, smart_vehicles):
        """
        Finds smart vehicles within a defined radius of the ego vehicle.
        """
        ego_transform = ego_vehicle.get_transform()
        vehicle_distances = {}

        for vehicle in smart_vehicles:
            try:
                # Use vehicle directly if it's already an actor
                vehicle_actor = vehicle if isinstance(vehicle, carla.Actor) else self.world.get_actor(vehicle)

                if vehicle_actor.id == ego_vehicle.id:
                    continue

                distance = ego_transform.location.distance(vehicle_actor.get_transform().location)

                if distance <= self.radius:
                    if vehicle_actor.id not in self.proximity_cache or \
                    not self.proximity_cache[vehicle_actor.id]['in_proximity']:
                        self.proximity_cache[vehicle_actor.id] = {'in_proximity': True, 'distance': distance}
                        logging.info(f"Smart Vehicle ID {vehicle_actor.id} is close to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m.")
                    vehicle_distances[vehicle_actor.id] = (vehicle_actor, distance)
                else:
                    if vehicle_actor.id in self.proximity_cache and self.proximity_cache[vehicle_actor.id]['in_proximity']:
                        self.proximity_cache[vehicle_actor.id] = {'in_proximity': False}
                        logging.info(f"Smart Vehicle ID {vehicle_actor.id} has exited proximity of Ego Vehicle.")

            except Exception as e:
                logging.error(f"Error processing vehicle {vehicle}: {e}")

        return vehicle_distances

    def find_lidar_sensor_id(self, vehicle_label, vehicle_mapping):
        """
        Locate the LIDAR ID for the vehicle.
        """
        sensors = vehicle_mapping.get(vehicle_label, {}).get("sensors", [])
        for sensor_id in sensors:
            try:
                if isinstance(sensor_id, carla.Actor):
                    sensor_id = sensor_id.id  # Extract ID if it's an actor
                # Fetch the sensor actor from the world
                sensor = self.world.get_actor(sensor_id)
                if sensor and 'lidar' in sensor.type_id:
                    return sensor_id
            except Exception as e:
                logging.error(f"Error retrieving sensor ID {sensor_id} for {vehicle_label}: {e}")
        return None  # Return None if no LIDAR sensor is found

    def send_data_to_ego(self, ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Sends the full pose data (position, rotation, speed, and LIDAR) from the smart vehicle to the ego vehicle.
        :param ego_address: Address of the ego vehicle.
        :param smart_vehicle_id: ID of the smart vehicle.
        :param smart_vehicle: Smart vehicle actor.
        :param vehicle_mapping: Dictionary containing vehicle and sensor mappings.
        :param lidar_data_buffer: Dictionary for buffering LIDAR data asynchronously.
        """
        transform = smart_vehicle.get_transform()
        position = {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z
        }
        rotation = {
            "roll": transform.rotation.roll,
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw
        }
        speed = smart_vehicle.get_velocity().length()

        # Find the corresponding vehicle label in vehicle_mapping
        vehicle_label = None
        for label, data in vehicle_mapping.items():
            if data.get("actor_id") == smart_vehicle_id:
                vehicle_label = label
                break

        if not vehicle_label:
            logging.error(f"Smart Vehicle ID {smart_vehicle_id} not found in vehicle_mapping.")
            return

        # Retrieve the LIDAR sensor ID dynamically
        lidar_sensor_id = self.find_lidar_sensor_id(vehicle_label, vehicle_mapping)
        data_compressor = DataCompressor()
        with lidar_data_lock:
            lidar_data = lidar_data_buffer.get(lidar_sensor_id, [])
        if lidar_data:
            if isinstance(lidar_data, memoryview):
                lidar_data = list(lidar_data)

            # Convert LiDAR data into NumPy format for visualization
            lidar_points = np.frombuffer(bytearray(lidar_data), dtype=np.float32).reshape(-1, 4)

            # Log the type of lidar_data before serialization
            # logging.info(f"Type of lidar_data before serialization: {type(lidar_data)}")
            logging.info(f"Sending LIDAR data from {vehicle_label} to Ego Vehicle: {len(lidar_data)} points.")
            # Log the original size of lidar_data
            original_size = len(json.dumps(lidar_data).encode('utf-8'))
            logging.info(f"Original LIDAR data size: {original_size} bytes.")
            # Compress the LIDAR data
            compressed_lidar_data = data_compressor.compress(lidar_data)
            if compressed_lidar_data:
                compressed_size = len(compressed_lidar_data.encode('utf-8'))
                logging.info(f"Compressed LIDAR data size: {compressed_size} bytes.")
                lidar_data = compressed_lidar_data
            else:
                logging.error(f"Failed to compress LIDAR data for {vehicle_label}.")
        else:
            logging.warning(f"No LIDAR data available for {vehicle_label}.")

        # Package smart vehicle data for transmission
        smart_data = {
            "id": smart_vehicle_id,
            "position": position,
            "rotation": rotation,
            "speed": speed,
            "lidar": lidar_data
        }

        # Send the data to the ego vehicle
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(ego_address)
                sock.sendall(json.dumps(smart_data).encode())
                logging.info(f"Data successfully sent from {vehicle_label} to Ego Vehicle.")
        except Exception as e:
            logging.error(f"Error sending data from {vehicle_label} to Ego Vehicle: {e}")
    
    def stream_data(self, ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Continuously sends data from a smart vehicle to the ego vehicle while in proximity.
        """
        try:
            while self.proximity_cache.get(smart_vehicle_id, {}).get('in_proximity', False):
                self.send_data_to_ego(
                    ego_address,
                    smart_vehicle_id,
                    smart_vehicle,
                    vehicle_mapping,
                    lidar_data_buffer,
                    lidar_data_lock
                )
                time.sleep(0.1)  # Adjust interval as needed
        except Exception as e:
            logging.error(f"Error during continuous data transmission for {smart_vehicle_id}: {e}")

    def log_proximity_and_trigger_communication(self, ego_vehicle, smart_vehicles, world, proximity_state, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Logs proximity of smart vehicles to the ego vehicle and triggers communication if necessary.
        """
        # Find vehicles in radius
        vehicles_in_radius = self.find_vehicles_in_radius(ego_vehicle, smart_vehicles)
        # print(vehicles_in_radius)
        ego_address = ('127.0.0.1', 65432)

        for smart_vehicle_id, (smart_vehicle, distance) in vehicles_in_radius.items():
            try:
                # Log and process newly detected vehicles
                if smart_vehicle_id not in proximity_state:
                    logging.info(
                        f"Smart Vehicle ID {smart_vehicle_id} is close to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m."
                    )
                    proximity_state[smart_vehicle_id] = True
                    self.proximity_cache[smart_vehicle_id] = {'in_proximity': True}
                    thread = Thread(
                        target=self.stream_data,
                        args=(ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock),
                        daemon=True
                    )
                    thread.start()
                    logging.info(f"Started streaming thread for Smart Vehicle ID {smart_vehicle_id}.")
                else:
                    logging.debug(
                        f"Smart Vehicle ID {smart_vehicle_id} is still within proximity of Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m."
                    )
            except Exception as e:
                logging.error(f"Error processing vehicle {smart_vehicle_id}: {e}")

        # Clean up proximity state for vehicles no longer in range
        for vid in list(proximity_state.keys()):
            if vid not in vehicles_in_radius:
                self.proximity_cache[vid] = {'in_proximity': False}
                proximity_state.pop(vid, None)
                logging.info(f"Smart Vehicle ID {vid} has exited proximity of the Ego Vehicle.")
