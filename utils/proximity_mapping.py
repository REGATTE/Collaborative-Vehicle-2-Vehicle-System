import logging
import carla
import socket
import json
from threading import Lock, Thread
from math import sqrt
import time


class ConnectionPool:
    """
    A thread-safe connection pool for managing reusable socket connections.
    Includes periodic cleanup to remove stale or idle connections.
    """

    def __init__(self, cleanup_interval=30, idle_timeout=60):
        """
        Initializes the connection pool.
        :param cleanup_interval: Time (in seconds) between cleanup runs.
        :param idle_timeout: Maximum idle time (in seconds) for a connection.
        """
        self.pool = {}
        self.lock = Lock()
        self.cleanup_interval = cleanup_interval
        self.idle_timeout = idle_timeout
        self.running = True

        # Start the cleanup thread
        self.cleanup_thread = Thread(target=self.cleanup, daemon=True)
        self.cleanup_thread.start()

    def get_connection(self, address):
        """
        Retrieves a reusable socket connection or creates a new one.
        :param address: Tuple containing the IP and port of the target server.
        :return: An active socket connection.
        """
        with self.lock:
            current_time = time.time()

            # Check if a valid connection exists in the pool
            if address in self.pool:
                conn, last_used = self.pool[address]
                if current_time - last_used < self.idle_timeout:
                    self.pool[address] = (conn, current_time)  # Update last used time
                    return conn
                else:
                    # Stale connection; close and remove it
                    conn.close()
                    del self.pool[address]

            # Create a new connection
            conn = socket.create_connection(address)
            self.pool[address] = (conn, current_time)
            return conn

    def cleanup(self):
        """
        Periodically removes stale or idle connections from the pool.
        """
        while self.running:
            time.sleep(self.cleanup_interval)
            current_time = time.time()
            with self.lock:
                stale_connections = [
                    address for address, (conn, last_used) in self.pool.items()
                    if current_time - last_used >= self.idle_timeout
                ]
                for address in stale_connections:
                    conn, _ = self.pool.pop(address)
                    conn.close()
                    logging.info(f"Closed stale connection to {address}.")

    def close_all(self):
        """
        Closes all connections and stops the cleanup thread.
        """
        self.running = False
        self.cleanup_thread.join()
        with self.lock:
            for conn, _ in self.pool.values():
                conn.close()
            self.pool.clear()
            logging.info("Closed all connections in the pool.")


class ProximityMapping:
    """
    A class for managing proximity-based operations in CARLA, such as detecting vehicles
    within a specified radius and communicating their data to an ego vehicle.
    """

    def __init__(self, world, radius=20.0):
        """
        Initializes the ProximityMapping class.
        :param world: CARLA world object representing the simulation environment.
        :param radius: Radius (in meters) to consider for proximity detection.
        """
        self.world = world
        self.radius = radius
        self.proximity_cache = {}  # Caches proximity states of vehicles
        self.connection_pool = ConnectionPool()  # Pool for managing socket connections
        logging.info(f"ProximityMapping initialized with radius: {self.radius} meters.")

    def calculate_distance(self, location1, location2):
        """
        Calculates the Euclidean distance between two CARLA locations.
        :param location1: First location object.
        :param location2: Second location object.
        :return: Distance (in meters) between the two locations.
        """
        return sqrt(
            (location1.x - location2.x) ** 2 +
            (location1.y - location2.y) ** 2 +
            (location1.z - location2.z) ** 2
        )

    def find_vehicles_in_radius(self, ego_vehicle, smart_vehicles):
        """
        Finds vehicles within the specified radius of the ego vehicle.
        :param ego_vehicle: The ego vehicle actor.
        :param smart_vehicles: List of smart vehicle actors to evaluate.
        :return: Dictionary of vehicles within proximity, with IDs as keys and tuples
                 (vehicle actor, distance) as values.
        """
        ego_transform = ego_vehicle.get_transform()
        vehicle_distances = {}

        for vehicle in smart_vehicles:
            try:
                vehicle_actor = self.world.get_actor(vehicle.id if isinstance(vehicle, carla.Actor) else vehicle)

                # Skip the ego vehicle itself
                if vehicle_actor.id == ego_vehicle.id:
                    continue

                # Calculate distance from ego vehicle
                distance = ego_transform.location.distance(vehicle_actor.get_transform().location)

                # Check if the vehicle is within the proximity radius
                if distance <= self.radius:
                    if vehicle_actor.id not in self.proximity_cache or \
                            not self.proximity_cache[vehicle_actor.id]['in_proximity']:
                        self.proximity_cache[vehicle_actor.id] = {'in_proximity': True, 'distance': distance}
                        logging.info(f"Vehicle ID {vehicle_actor.id} is close to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m.")
                    vehicle_distances[vehicle_actor.id] = (vehicle_actor, distance)
                else:
                    # Update cache if the vehicle exits the proximity radius
                    if vehicle_actor.id in self.proximity_cache and self.proximity_cache[vehicle_actor.id]['in_proximity']:
                        self.proximity_cache[vehicle_actor.id] = {'in_proximity': False}
                        logging.info(f"Vehicle ID {vehicle_actor.id} exited proximity of Ego Vehicle.")

            except Exception as e:
                logging.error(f"Error processing vehicle {vehicle}: {e}")

        return vehicle_distances

    def get_vehicle_transform_data(self, smart_vehicle):
        """
        Retrieves the position, rotation, and speed of the smart vehicle.
        :param smart_vehicle: Smart vehicle actor.
        :return: Tuple containing position, rotation, and speed data.
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
        return position, rotation, speed

    def get_vehicle_label(self, smart_vehicle_id, vehicle_mapping):
        """
        Finds the vehicle label for a given smart vehicle ID.
        :param smart_vehicle_id: ID of the smart vehicle.
        :param vehicle_mapping: Mapping of vehicle labels to metadata.
        :return: Vehicle label as a string, or None if not found.
        """
        return next(
            (label for label, data in vehicle_mapping.items() if data.get("actor_id") == smart_vehicle_id),
            None
        )

    def get_lidar_data(self, vehicle_label, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Retrieves LIDAR data for the specified vehicle.
        :param vehicle_label: Label of the vehicle in the mapping.
        :param vehicle_mapping: Mapping of vehicle labels to metadata.
        :param lidar_data_buffer: Buffer containing LIDAR data.
        :param lidar_data_lock: Lock for thread-safe access to the buffer.
        :return: List of LIDAR data points, or an empty list if no data is available.
        """
        lidar_sensor_id = next(
            (sensor_id for sensor_id in vehicle_mapping.get(vehicle_label, {}).get("sensors", [])
             if 'lidar' in vehicle_mapping[vehicle_label].get("type_id", "")),
            None
        )
        with lidar_data_lock:
            lidar_data = lidar_data_buffer.get(lidar_sensor_id, [])
        return lidar_data

    def send_smart_vehicle_data(self, conn, smart_vehicle_id, position, rotation, speed, lidar_data, vehicle_label):
        """
        Sends the packaged data of a smart vehicle to the ego vehicle.
        :param conn: Socket connection to the ego vehicle.
        :param smart_vehicle_id: ID of the smart vehicle.
        :param position: Position data of the vehicle.
        :param rotation: Rotation data of the vehicle.
        :param speed: Speed of the vehicle.
        :param lidar_data: LIDAR data of the vehicle.
        :param vehicle_label: Label of the vehicle.
        """
        smart_data = {
            "id": smart_vehicle_id,
            "position": position,
            "rotation": rotation,
            "speed": speed,
            "lidar": lidar_data
        }
        conn.sendall(json.dumps(smart_data).encode())

    def send_data_to_ego(self, ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Orchestrates the process of sending smart vehicle data to the ego vehicle.
        :param ego_address: Address (IP, port) of the ego vehicle.
        :param smart_vehicle_id: ID of the smart vehicle.
        :param smart_vehicle: Smart vehicle actor.
        :param vehicle_mapping: Mapping of vehicle labels to metadata.
        :param lidar_data_buffer: Buffer for LIDAR data.
        :param lidar_data_lock: Lock for thread-safe access to the buffer.
        """
        position, rotation, speed = self.get_vehicle_transform_data(smart_vehicle)
        vehicle_label = self.get_vehicle_label(smart_vehicle_id, vehicle_mapping)
        if not vehicle_label:
            logging.error(f"Smart Vehicle ID {smart_vehicle_id} not found in vehicle_mapping.")
            return
        lidar_data = self.get_lidar_data(vehicle_label, vehicle_mapping, lidar_data_buffer, lidar_data_lock)
        try:
            conn = self.connection_pool.get_connection(ego_address)
            self.send_smart_vehicle_data(conn, smart_vehicle_id, position, rotation, speed, lidar_data, vehicle_label)
        except Exception as e:
            logging.error(f"Error in connection for {vehicle_label}: {e}")

    def log_proximity_and_trigger_communication(self, ego_vehicle, smart_vehicles, world, proximity_state, vehicle_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Monitors proximity of smart vehicles to the ego vehicle and triggers communication.
        :param ego_vehicle: Ego vehicle actor.
        :param smart_vehicles: List of smart vehicle actors to monitor.
        :param world: CARLA world object.
        :param proximity_state: Dictionary tracking the proximity state of vehicles.
        :param vehicle_mapping: Mapping of vehicle labels to metadata.
        :param lidar_data_buffer: Buffer for LIDAR data.
        :param lidar_data_lock: Lock for thread-safe access to the buffer.
        :return: Updated proximity state dictionary.
        """
        vehicles_in_radius = self.find_vehicles_in_radius(ego_vehicle, smart_vehicles)
        ego_address = ('127.0.0.1', 65432)
        for smart_vehicle_id, (smart_vehicle, distance) in vehicles_in_radius.items():
            try:
                if smart_vehicle_id not in proximity_state:
                    logging.info(f"Vehicle ID {smart_vehicle_id} is close to Ego Vehicle (ID: {ego_vehicle.id}) at {distance:.2f}m.")
                    proximity_state[smart_vehicle_id] = True
                    self.send_data_to_ego(
                        ego_address, smart_vehicle_id, smart_vehicle, vehicle_mapping, lidar_data_buffer, lidar_data_lock
                    )
            except Exception as e:
                logging.error(f"Error processing vehicle {smart_vehicle_id}: {e}")
        proximity_state = {vid: state for vid, state in proximity_state.items() if vid in vehicles_in_radius}
        return proximity_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: Main section to initialize CARLA and use ProximityMapping
