import socket
import threading
import logging
import json
import numpy as np
from utils.proximity_mapping import ProximityMapping


class EgoVehicleListener:
    def __init__(self, host='127.0.0.1', port=65432, ego_vehicle=None, world=None, vehicle_mapping=None):
        """
        Initializes the ego vehicle listener.
        :param host: Host address for the listener.
        :param port: Port for the listener.
        :param ego_vehicle: Ego vehicle actor.
        :param world: CARLA world instance.
        :param vehicle_mapping: Vehicle mapping dictionary.
        """
        self.host = host
        self.port = port
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.vehicle_mapping = vehicle_mapping
        self.lidar_data_proximity = {}  # Store LIDAR data for vehicles in proximity
        self.proximity_mapping = ProximityMapping(world, radius=20.0)  # Use proximity mapping
        self.running = True  # Flag to safely terminate thread

    def compute_relative_pose(self, smart_data):
        """
        Computes the pose of the smart vehicle relative to the ego vehicle.
        :param smart_data: Data received from the smart vehicle (global pose).
        :return: Dictionary with relative position and rotation.
        """
        smart_position = np.array([
            smart_data['position']['x'],
            smart_data['position']['y'],
            smart_data['position']['z']
        ])
        ego_transform = self.ego_vehicle.get_transform()
        ego_position = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.location.z
        ])

        # Extract yaw angles
        smart_yaw = smart_data['rotation']['yaw']
        ego_yaw = ego_transform.rotation.yaw

        # Compute relative position
        delta_position = smart_position - ego_position
        ego_yaw_rad = np.radians(ego_yaw)
        rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), np.sin(ego_yaw_rad), 0],
            [-np.sin(ego_yaw_rad), np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])
        relative_position = np.dot(rotation_matrix, delta_position)

        # Compute relative yaw
        relative_yaw = smart_yaw - ego_yaw

        return {
            "relative_position": {
                "x": relative_position[0],
                "y": relative_position[1],
                "z": relative_position[2]
            },
            "relative_yaw": relative_yaw
        }
    
    def transform_lidar_points(self, lidar_points, relative_position, relative_yaw):
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
        combined_lidar = []
        for vehicle_id, lidar_points in self.lidar_data_proximity.items():
            relative_pose = self.compute_relative_pose(self.vehicle_mapping[vehicle_id])
            relative_position = relative_pose['relative_position']
            relative_yaw = relative_pose['relative_yaw']
            transformed_points = self.transform_lidar_points(lidar_points, relative_position, relative_yaw)
            combined_lidar.extend(transformed_points)

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
        try:
            data = conn.recv(4096)
            if data:
                smart_data = json.loads(data.decode())
                smart_vehicle_id = smart_data['id']

                vehicles_in_radius = self.proximity_mapping.find_vehicles_in_radius(
                    self.ego_vehicle, 
                    [self.world.get_actor(data["actor_id"]) for data in self.vehicle_mapping.values()]
                )

                if smart_vehicle_id in vehicles_in_radius:
                    logging.info(f"Received data from nearby Smart Vehicle {smart_vehicle_id}:")
                    logging.info(f"  Global Position: {smart_data['position']}")
                    logging.info(f"  Global Rotation: {smart_data['rotation']}")
                    logging.info(f"  Speed: {smart_data['speed']:.2f} m/s")

                    if "lidar" in smart_data:
                        lidar_points = smart_data['lidar']
                        logging.info(f"  LIDAR Data: {len(lidar_points)} points received.")
                        self.lidar_data_proximity[smart_vehicle_id] = lidar_points

                    combined_lidar = self.combine_lidar_data()
                    logging.info(f"Path planning can now use combined LIDAR data with {len(combined_lidar)} points.")
                else:
                    logging.debug(f"Smart Vehicle {smart_vehicle_id} is not in proximity. Ignoring data.")
        except Exception as e:
            logging.error(f"Error in connection: {e}")
        finally:
            conn.close()

    def stop_listener(self):
        """
        Stops the listener gracefully.
        """
        self.running = False
