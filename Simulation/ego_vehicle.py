import socket
import threading
import logging
import json
import numpy as np

from utils.proximity_mapping import ProximityMapping
from utils.vehicle_mapping.vehicle_mapping import load_vehicle_mapping

class EgoVehicleListener:
    def __init__(self, host='127.0.0.1', port=65432, ego_vehicle=None, world=None):
        """
        Initializes the ego vehicle listener.
        :param host: Host address for the listener.
        :param port: Port for the listener.
        :param ego_vehicle: Ego vehicle actor.
        :param world: CARLA world instance.
        :param vehicle_mapping: Vehicle mapping dictionary.
        """
        if world is None:
            raise ValueError("CARLA world instance is required.")
        self.host = host
        self.port = port
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.vehicle_mapping = load_vehicle_mapping()
        self.lidar_data_proximity = {}  # Store LIDAR data for vehicles in proximity
        self.proximity_mapping = ProximityMapping(world, radius=20.0)  # Use proximity mapping
        self.running = True  # Flag to safely terminate thread
    
    def get_vehicle_id(self, label):
        """
        Retrieve the actor ID for a given vehicle label.
        """
        return self.vehicle_mapping.get(label, {}).get("actor_id")
    
    def get_vehicle_label(self, actor_id):
        """
        Retrieve the label for a given actor ID.
        """
        return next(
            (
                label
                for label, data in self.vehicle_mapping.items()
                if data.get("actor_id") == actor_id
            ),
            None,
        )

    def compute_relative_pose(self, vehicle_label):
        """
        Computes the pose of the smart vehicle relative to the ego vehicle.
        :param vehicle_label: Label of the smart vehicle.
        :return: Dictionary with relative position and rotation.
        """
        vehicle_data = self.vehicle_mapping.get(vehicle_label)
        if not vehicle_data:
            raise ValueError(f"Vehicle label {vehicle_label} not found in mapping.")

        vehicle_position = np.array([
            vehicle_data["initial_position"]["x"],
            vehicle_data["initial_position"]["y"],
            vehicle_data["initial_position"]["z"]
        ])
        ego_transform = self.ego_vehicle.get_transform()
        ego_position = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.location.z
        ])

        # Extract yaw angles
        vehicle_yaw = vehicle_data.get("rotation", {}).get("yaw", 0)  # Default to 0 if yaw is missing
        ego_yaw = ego_transform.rotation.yaw

        # Compute relative position
        delta_position = vehicle_position - ego_position
        ego_yaw_rad = np.radians(ego_yaw)
        rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), np.sin(ego_yaw_rad), 0],
            [-np.sin(ego_yaw_rad), np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])
        relative_position = np.dot(rotation_matrix, delta_position)

        # Compute relative yaw
        relative_yaw = vehicle_yaw - ego_yaw

        return {
            "relative_position": {
                "x": relative_position[0],
                "y": relative_position[1],
                "z": relative_position[2]
            },
            "relative_yaw": relative_yaw
        }
    
    def transform_lidar_points(self, lidar_points, relative_position, relative_yaw):
        """
        Transforms LIDAR points from a smart vehicle to the ego vehicle's coordinate frame.
        """
        relative_yaw_rad = np.radians(relative_yaw)
        rotation_matrix = np.array([
            [np.cos(relative_yaw_rad), -np.sin(relative_yaw_rad), 0],
            [np.sin(relative_yaw_rad),  np.cos(relative_yaw_rad), 0],
            [0, 0, 1]
        ])

        # Convert lidar points to a NumPy array
        lidar_points = np.array(lidar_points)

        # Apply rotation matrix to all points
        rotated_points = np.dot(lidar_points, rotation_matrix.T)

        return rotated_points + np.array(
            [
                relative_position['x'],
                relative_position['y'],
                relative_position['z'],
            ]
        )
    
    def combine_lidar_data(self):
        """
        Combines LIDAR data from all nearby smart vehicles into a single dataset.
        """
        combined_lidar = []
        for vehicle_label, lidar_points in self.lidar_data_proximity.items():
            if vehicle_label not in self.vehicle_mapping:
                logging.warning(f"Vehicle label {vehicle_label} not found in vehicle mapping. Skipping.")
                continue

            try:
                relative_pose = self.compute_relative_pose(vehicle_label)
                relative_position = relative_pose["relative_position"]
                relative_yaw = relative_pose["relative_yaw"]
                transformed_points = self.transform_lidar_points(lidar_points, relative_position, relative_yaw)
                combined_lidar.extend(transformed_points)
            except Exception as e:
                logging.error(f"Error processing LIDAR data for vehicle {vehicle_label}: {e}")

        logging.info(f"Combined LIDAR Data: {len(combined_lidar)} points across all nearby vehicles.")
        return combined_lidar

    def start_listener(self):
        """
        Starts the listener to receive data from smart vehicles and compute relative pose.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        logging.info(f"Ego Vehicle listening on {self.host}:{self.port}")

        while self.running:
            conn, addr = server_socket.accept()
            threading.Thread(target=self.handle_connection, args=(conn,)).start()

        server_socket.close()

    def handle_connection(self, conn):
        """
        Handles incoming connections from smart vehicles.
        """
        try:
            data = conn.recv(4096)
            if not data:
                logging.warning("Received empty data. Ignoring connection.")
                return

            smart_data = json.loads(data.decode())
            smart_vehicle_id = smart_data['id']
            vehicle_label = self.get_vehicle_label(smart_vehicle_id)

            if not vehicle_label:
                logging.warning(f"Smart Vehicle ID {smart_vehicle_id} not found in mapping.")
                return

            logging.info(f"Received data from Smart Vehicle {vehicle_label} (ID: {smart_vehicle_id}).")

            # Build a list of valid actors for proximity mapping
            actors = []
            for label, data in self.vehicle_mapping.items():
                actor_id = data.get("actor_id")
                if actor_id is None:
                    # logging.warning(f"No actor_id found for vehicle {label}. Skipping.")
                    continue

                actor = self.world.get_actor(actor_id)
                if actor is None:
                    logging.warning(f"Actor with ID {actor_id} (label: {label}) not found in CARLA world.")
                else:
                    logging.info(f"Actor with ID {actor_id} found: {actor.type_id}")
                    logging.debug(f"Actor Details: {actor.attributes}")  # Log actor attributes for additional details
                    actors.append(actor)

            if not actors:
                logging.error("No valid actors found in the CARLA world.")
                return

            vehicles_in_radius = self.proximity_mapping.find_vehicles_in_radius(self.ego_vehicle, actors)
            # Log the vehicles in radius
            if vehicles_in_radius:
                logging.info(f"Vehicles in radius: {vehicles_in_radius}")
            else:
                logging.info("No vehicles in proximity of the Ego Vehicle.")

            if smart_vehicle_id not in vehicles_in_radius:
                logging.debug(f"Smart Vehicle {vehicle_label} is not in proximity. Ignoring data.")
                return

            self._process_vehicle_data(vehicle_label, smart_data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON data: {e}")
        except Exception as e:
            logging.error(f"Error in connection: {e}")
        finally:
            conn.close()

    def _process_vehicle_data(self, vehicle_label, smart_data):
        """
        Processes data received from a smart vehicle.
        """
        logging.info(f"Processing data from Smart Vehicle {vehicle_label}:")

        lidar_points = self.lidar_data_proximity.get(vehicle_label, [])
        if not lidar_points:
            logging.warning(f"No LIDAR data available for Smart Vehicle {vehicle_label}.")
            return

        logging.info(f"  LIDAR Data: {len(lidar_points)} points received.")
        self.lidar_data_proximity[vehicle_label] = lidar_points

        try:
            combined_lidar = self.combine_lidar_data()
            logging.info(f"Path planning can now use combined LIDAR data with {len(combined_lidar)} points.")
        except Exception as e:
            logging.error(f"Error combining LIDAR data: {e}")

    def stop_listener(self):
        """
        Stops the listener gracefully.
        """
        self.running = False

