import logging
import carla
import socket
import json
from threading import Lock
from math import sqrt
import time
import threading

class ConnectionPool:
    def __init__(self):
        self.pool = {}
        self.lock = Lock()
        self.last_used = {}
        self.cleanup_thread = None

    def get_connection(self, address):
        with self.lock:
            if address in self.pool:
                return self.pool[address]
            else:
                # Create a new connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(address)
                self.pool[address] = sock
                self.last_used[address] = time.time()
                return sock

    def close_connection(self, address):
        with self.lock:
            if address in self.pool:
                self.pool[address].close()
                del self.pool[address]
    
    def cleanup_idle_connections(self, timeout=300):
        current_time = time.time()
        with self.lock:
            for address in list(self.pool.keys()):
                if current_time - self.last_used[address] > timeout:
                    logging.info(f"Closing idle connection to {address} after {timeout} seconds of inactivity.")
                    self.close_connection(address)

    def close_all(self):
        with self.lock:
            for sock in self.pool.values():
                sock.close()
            self.pool.clear()

    def start_periodic_cleanup(self, interval=60, timeout=300):
        """
        Starts a background thread to periodically clean up idle connections.
        :param interval: Time interval between cleanup runs (in seconds).
        :param timeout: Idle time (in seconds) before a connection is considered idle.
        """
        def cleanup_task():
            while True:
                time.sleep(interval)
                logging.info("Running periodic cleanup of idle connections.")
                self.cleanup_idle_connections(timeout)

        if not self.cleanup_thread or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
            self.cleanup_thread.start()

connection_pool = ConnectionPool()
# Start periodic cleanup with an interval of 60 seconds and a timeout of 300 seconds for idle connections
connection_pool.start_periodic_cleanup(interval=60, timeout=300)

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
                vehicle_actor = self.world.get_actor(vehicle.id if isinstance(vehicle, carla.Actor) else vehicle)

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
        Improved logic to locate the LIDAR ID for the vehicle.
        """
        sensors = vehicle_mapping.get(vehicle_label, {}).get("sensors", [])
        for sensor_id in sensors:
            # Match against known LIDAR sensor types or specific IDs
            sensor = self.world.get_actor(sensor_id)
            if sensor and 'lidar' in sensor.type_id:
                return sensor_id
        return None

    def send_data_to_ego(self, ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock, max_retries=5, retry_delay=2):
        """
        Sends the full pose data (position, rotation, speed, and LIDAR) from the smart vehicle to the ego vehicle.
        Implements a retry mechanism to handle transient failures.
        :param ego_address: Address of the ego vehicle.
        :param smart_vehicle_id: ID of the smart vehicle.
        :param smart_vehicle: Smart vehicle actor.
        :param vehicle_mapping: Dictionary containing vehicle and sensor mappings.
        :param lidar_data_buffer: Dictionary for buffering LIDAR data asynchronously.
        :param lidar_data_lock: Lock object for thread-safe access to LIDAR data.
        :param max_retries: Maximum number of retries before giving up.
        :param retry_delay: Delay between retries in seconds.
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

        with lidar_data_lock:
            lidar_data = lidar_data_buffer.get(lidar_sensor_id, [])
        if lidar_data:
            logging.info(f"Sending LIDAR data from {vehicle_label} to Ego Vehicle: {len(lidar_data)} points.")
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

        # Retry mechanism
        for attempt in range(1, max_retries + 1):
            try:
                sock = connection_pool.get_connection(ego_address)
                sock.sendall(json.dumps(smart_data).encode())
                logging.info(f"Data successfully sent to Ego Vehicle at {ego_address} on attempt {attempt}.")
                return  # Exit on success
            except Exception as e:
                logging.warning(f"Attempt {attempt} failed to send data to Ego Vehicle: {e}")
                connection_pool.close_connection(ego_address)  # Close problematic connection
                if attempt < max_retries:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All {max_retries} attempts to send data to Ego Vehicle failed. Giving up.")

        # Fallback action if retries fail
        logging.error(f"Failed to send data for {vehicle_label} to Ego Vehicle after {max_retries} attempts.")

    def log_proximity_and_trigger_communication(self, ego_vehicle, smart_vehicles, world, proximity_state, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Logs proximity of smart vehicles to the ego vehicle and triggers communication if necessary.
        """
        # Find vehicles in radius
        vehicles_in_radius = self.find_vehicles_in_radius(ego_vehicle, smart_vehicles)
        ego_address = ('127.0.0.1', 65432)

        for smart_vehicle_id, (smart_vehicle, distance) in vehicles_in_radius.items():
            try:
                # Log and process newly detected vehicles
                if smart_vehicle_id not in proximity_state:
                    logging.info(
                        f"Smart Vehicle ID {smart_vehicle_id} is close to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m."
                    )
                    proximity_state[smart_vehicle_id] = True
                    self.send_data_to_ego(
                        ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock
                    )
                else:
                    logging.debug(
                        f"Smart Vehicle ID {smart_vehicle_id} is still within proximity of Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m."
                    )
            except Exception as e:
                logging.error(f"Error processing vehicle {smart_vehicle_id}: {e}")

        # Clean up proximity state for vehicles no longer in range
        proximity_state = {
            vid: state for vid, state in proximity_state.items()
            if vid in vehicles_in_radius
        }

        return proximity_state
