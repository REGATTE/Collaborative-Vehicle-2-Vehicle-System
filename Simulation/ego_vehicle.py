import socket
import logging
import json
import numpy as np

class EgoVehicleListener:
    def __init__(self, host='127.0.0.1', port=65432, ego_vehicle=None):
        """
        Initializes the ego vehicle listener.
        :param host: Host address for the listener.
        :param port: Port for the listener.
        :param ego_vehicle: Ego vehicle actor.
        """
        self.host = host
        self.port = port
        self.ego_vehicle = ego_vehicle

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

    def start_listener(self):
        """
        Starts the listener to receive data from smart vehicles and compute relative pose.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        logging.info(f"Ego Vehicle listening on {self.host}:{self.port}")

        while True:
            conn, addr = server_socket.accept()
            try:
                data = conn.recv(1024)
                if data:
                    smart_data = json.loads(data.decode())
                    logging.info(f"Received data from Smart Vehicle {smart_data['id']}:")
                    logging.info(f"  Global Position: {smart_data['position']}")
                    logging.info(f"  Global Rotation: {smart_data['rotation']}")
                    logging.info(f"  Speed: {smart_data['speed']:.2f} m/s")
                    
                    # Compute relative pose if ego vehicle is defined
                    if self.ego_vehicle:
                        relative_pose = self.compute_relative_pose(smart_data)
                        logging.info(f"  Relative Position: {relative_pose['relative_position']}")
                        logging.info(f"  Relative Yaw: {relative_pose['relative_yaw']:.2f} degrees")
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
            finally:
                conn.close()
