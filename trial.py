import numpy as np

class VehicleTransformation:
    def __init__(self, ego_position, ego_yaw, smart_position, smart_yaw):
        """
        Initialize the transformation parameters.

        :param ego_position: List or array [x, y, z] for the ego vehicle's global position.
        :param ego_yaw: Ego vehicle's yaw in degrees.
        :param smart_position: List or array [x, y, z] for the smart vehicle's global position.
        :param smart_yaw: Smart vehicle's yaw in degrees.
        """
        self.ego_position = np.array(ego_position)
        self.ego_yaw = ego_yaw
        self.smart_position = np.array(smart_position)
        self.smart_yaw = smart_yaw

    def transform_lidar_points(self, lidar_points):
        """
        Transform LIDAR points from the smart vehicle's frame to the ego vehicle's frame.

        :param lidar_points: NumPy array of LIDAR points in XYZI format (N x 4).
        :return: Transformed LIDAR points in XYZI format.
        """
        # Convert yaw angles to radians
        smart_yaw_rad = np.radians(self.smart_yaw)
        ego_yaw_rad = np.radians(self.ego_yaw)

        # Compute relative yaw
        relative_yaw_rad = smart_yaw_rad - ego_yaw_rad

        # Compute delta position (translation vector)
        delta_position = self.smart_position - self.ego_position

        # Rotation matrix for transforming points to the ego's coordinate frame
        ego_rotation_matrix = np.array([
            [np.cos(-ego_yaw_rad), -np.sin(-ego_yaw_rad), 0],
            [np.sin(-ego_yaw_rad),  np.cos(-ego_yaw_rad), 0],
            [0,                    0,                    1]
        ])

        # Rotate the delta position into the ego's frame
        relative_position = np.dot(ego_rotation_matrix, delta_position)

        # Rotation matrix for the relative yaw (smart vehicle's points)
        relative_rotation_matrix = np.array([
            [np.cos(relative_yaw_rad), -np.sin(relative_yaw_rad), 0],
            [np.sin(relative_yaw_rad),  np.cos(relative_yaw_rad), 0],
            [0,                        0,                        1]
        ])

        # Extract XYZ and intensity from LIDAR points
        lidar_points = np.array(lidar_points)
        xyz_points = lidar_points[:, :3]
        intensity = lidar_points[:, 3]

        # Rotate LIDAR points from the smart vehicle to the ego vehicle frame
        rotated_points = np.dot(xyz_points, relative_rotation_matrix.T)
        transformed_points = rotated_points + relative_position

        # Combine transformed XYZ with intensity
        return np.hstack((transformed_points, intensity.reshape(-1, 1)))

# Example usage
if __name__ == "__main__":
    # Global positions and rotations
    ego_position = [5, 10, 0]  # Ego vehicle global position [x, y, z]
    ego_yaw = 30.0  # Ego vehicle yaw in degrees
    smart_position = [10, 15, 0]  # Smart vehicle global position [x, y, z]
    smart_yaw = 45.0  # Smart vehicle yaw in degrees

    # Example LIDAR points in the smart vehicle's frame
    lidar_points = np.array([
        [1, 2, 0, 0.8],
        [3, 4, 0, 0.5],
        [5, 6, 0, 0.9]
    ])  # Format: [x, y, z, intensity]

    # Create the transformation object
    transformer = VehicleTransformation(ego_position, ego_yaw, smart_position, smart_yaw)

    # Transform the LIDAR points
    transformed_points = transformer.transform_lidar_points(lidar_points)

    print("Transformed LIDAR Points:")
    print(transformed_points)
