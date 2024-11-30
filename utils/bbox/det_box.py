import numpy as np
import json
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
import time
import logging
import cv2

class LidarBoundingBoxDetector:
    def __init__(self, output_dir="combined_bounding_det_boxes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def cluster_point_cloud(self, point_cloud, eps=0.5, min_samples=10):
        """
        Cluster the combined point cloud using DBSCAN.

        :param point_cloud: Numpy array of shape (N, 3) containing point cloud data.
        :param eps: Maximum distance between points to consider them in the same cluster.
        :param min_samples: Minimum number of points required to form a cluster.
        :return: List of clusters, each cluster is a numpy array of points.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud[:, :3])
        labels = clustering.labels_

        # Extract clusters
        clusters = []
        for label in set(labels):
            if label == -1:  # Ignore noise points
                continue
            cluster_points = point_cloud[labels == label, :3]
            clusters.append(cluster_points)

        return clusters

    def fit_bounding_box(self, cluster):
        """
        Fit an axis-aligned bounding box to a cluster of points.

        :param cluster: Numpy array of shape (N, 3) representing a cluster of points.
        :return: Dictionary with bounding box extents and location.
        """
        x_min, y_min, z_min = cluster.min(axis=0)
        x_max, y_max, z_max = cluster.max(axis=0)

        location = {
            "x": (x_min + x_max) / 2,
            "y": (y_min + y_max) / 2,
            "z": (z_min + z_max) / 2,
        }
        extent = {
            "x": (x_max - x_min) / 2,
            "y": (y_max - y_min) / 2,
            "z": (z_max - z_min) / 2,
        }

        return {
            "location": location,
            "extent": extent,
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        }

    def detect_bounding_boxes(self, point_cloud, eps=0.5, min_samples=10):
        """
        Detect bounding boxes from combined LiDAR data.

        :param point_cloud: Numpy array of shape (N, 3) or (N, 4) containing combined point cloud data.
        :param eps: DBSCAN clustering distance parameter.
        :param min_samples: DBSCAN minimum samples parameter.
        :return: List of bounding box dictionaries.
        """
        clusters = self.cluster_point_cloud(point_cloud, eps=eps, min_samples=min_samples)
        bounding_boxes = []
        for cluster in clusters:
            bbox = self.fit_bounding_box(cluster)
            bounding_boxes.append(bbox)

        return bounding_boxes
    
    def transform_bounding_boxes_to_ego_frame(self, bounding_boxes, ego_location, ego_yaw):
        """
        Transforms detected bounding boxes to the ego vehicle's coordinate frame.

        :param bounding_boxes: List of detected bounding boxes.
        :param ego_location: Dictionary with keys 'x', 'y', 'z' for the ego vehicle's position.
        :param ego_yaw: Yaw of the ego vehicle in degrees.
        :return: List of transformed bounding boxes.
        """
        ego_yaw_rad = np.radians(ego_yaw)
        rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), -np.sin(ego_yaw_rad), 0],
            [np.sin(ego_yaw_rad),  np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])

        transformed_bounding_boxes = []

        for bbox in bounding_boxes:
            # Extract the location of the bounding box
            bbox_location = np.array([bbox['location']['x'], bbox['location']['y'], bbox['location']['z']])
            
            # Compute the relative position
            relative_position = bbox_location - np.array([ego_location['x'], ego_location['y'], ego_location['z']])
            
            # Apply the rotation matrix to transform to ego frame
            transformed_position = np.dot(rotation_matrix, relative_position)

            # Update the bounding box location
            bbox['location'] = {
                "x": transformed_position[0],
                "y": transformed_position[1],
                "z": transformed_position[2]
            }
            
            # Adjust yaw relative to the ego vehicle
            bbox['rotation']['yaw'] -= ego_yaw

            transformed_bounding_boxes.append(bbox)

        return transformed_bounding_boxes
    
    def detect_and_transform_bounding_boxes(self, point_cloud, ego_location, ego_yaw, eps=0.5, min_samples=10):
        """
        Detect bounding boxes from the point cloud and transform them to the ego vehicle's frame.

        :param point_cloud: Combined LiDAR point cloud data.
        :param ego_location: Dictionary with keys 'x', 'y', 'z' for the ego vehicle's position.
        :param ego_yaw: Yaw of the ego vehicle in degrees.
        :param eps: DBSCAN clustering distance parameter.
        :param min_samples: DBSCAN minimum samples parameter.
        :return: List of transformed bounding boxes.
        """
        # Detect bounding boxes
        detected_bounding_boxes = self.detect_bounding_boxes(point_cloud, eps, min_samples)
        
        # Transform bounding boxes to the ego frame
        transformed_bounding_boxes = self.transform_bounding_boxes_to_ego_frame(
            detected_bounding_boxes, ego_location, ego_yaw
        )
        
        return transformed_bounding_boxes

    def save_bounding_boxes(self, bounding_boxes, output_dir="det_bounding_boxes"):
        """
        Saves bounding boxes to a JSON file.

        :param bounding_boxes: List of bounding box dictionaries.
        :param output_dir: Directory to save the bounding boxes JSON file.
        :return: The name of the saved bounding boxes file.
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate a timestamped filename
            filename = f"det_bounding_boxes_{int(time.time() * 1000)}.json"
            filepath = os.path.join(output_dir, filename)

            # Save bounding boxes to the JSON file
            with open(filepath, "w") as f:
                json.dump(bounding_boxes, f, indent=4)

            logging.info(f"Det Bounding boxes saved to: {filepath}")
            return filename
        except Exception as e:
            logging.error(f"Failed to save det bounding boxes: {e}")
            return None
    
    def compute_bounding_box_3d_lidar(self, location, extent, rotation, ego_location):
        """
        Compute the 8 corners of a 3D bounding box based on its location, extent, rotation,
        and relative to the ego vehicle's position.

        :param location: Dictionary with keys 'x', 'y', 'z' for the center of the bounding box.
        :param extent: Dictionary with keys 'x', 'y', 'z' for the half-width, half-length, and half-height.
        :param rotation: Dictionary with key 'yaw' (rotation in degrees around the Z-axis).
        :param ego_location: Dictionary with keys 'x', 'y', 'z' for the ego vehicle's position.
        :return: NumPy array of shape (8, 3), representing the 3D coordinates of the corners relative to the ego vehicle.
        """
        # Compute the bounding box corners in the world frame
        x, y, z = extent['x'], extent['y'], extent['z']
        corners = np.array([
            [x, y, z],  [x, -y, z],  [-x, -y, z],  [-x, y, z],  # Top four corners
            [x, y, -z], [x, -y, -z], [-x, -y, -z], [-x, y, -z]  # Bottom four corners
        ])

        # Create a rotation matrix for the yaw angle
        yaw_rad = np.radians(rotation['yaw'])
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0,                0,               1]
        ])

        # Rotate the corners to the world frame
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate the rotated corners to the global position
        location_vec = np.array([location['x'], location['y'], location['z']])
        world_corners = rotated_corners + location_vec

        # Shift corners to be relative to the ego vehicle
        ego_vec = np.array([ego_location['x'], ego_location['y'], ego_location['z']])
        return world_corners - ego_vec

    def project_bbox_to_2d_lidar(self, corners_3d, frame_size, lidar_range):
        """
        Project 3D bounding box corners to 2D pixel coordinates based on the LiDAR frame.

        :param corners_3d: NumPy array of shape (8, 3) representing the 3D corners relative to the ego vehicle.
        :param frame_size: Tuple (width, height) for the frame.
        :param lidar_range: Maximum range of the LiDAR sensor.
        :return: Tuple (x_min, y_min, x_max, y_max) representing the bounding box in pixel coordinates.
        """
        width, height = frame_size
        scale_x = width / lidar_range
        scale_y = height / lidar_range

        # Map corners to pixel coordinates
        corners_2d = []
        for corner in corners_3d:
            x, y, _ = corner
            px = int((x + lidar_range / 2) * scale_x)
            py = int((lidar_range / 2 - y) * scale_y)
            corners_2d.append((px, py))

        # Compute bounding box limits
        corners_2d = np.array(corners_2d)
        x_min, y_min = np.min(corners_2d, axis=0)
        x_max, y_max = np.max(corners_2d, axis=0)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def plot_bounding_boxes_on_lidar_frame(self, frame_path, bounding_boxes, frame_size, lidar_range, ego_location, output_dir="frames_with_bboxes"):
        """
        Plots bounding boxes on a saved LiDAR frame and saves the updated frame.

        :param frame_path: Path to the LiDAR frame image (PNG).
        :param bounding_boxes: List of bounding box dictionaries with 'location', 'extent', 'rotation'.
        :param frame_size: Tuple of (width, height) for the frame.
        :param lidar_range: Maximum range of the LiDAR sensor.
        :param ego_location: Dictionary with 'x', 'y', 'z' for the ego vehicle's location.
        :param output_dir: Directory to save the updated frames with bounding boxes.
        """
        try:
            logging.info(f"Loading LiDAR frame from: {frame_path}")
            # Load the frame
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load frame: {frame_path}")
            logging.info(f"LiDAR frame loaded successfully. Frame size: {frame.shape}")

            # Frame size dimensions
            frame_width, frame_height = frame_size
            logging.info(f"Frame dimensions: width={frame_width}, height={frame_height}")

            # Process bounding boxes
            for idx, bbox in enumerate(bounding_boxes):
                try:
                    logging.info(f"Processing bounding box {idx + 1}/{len(bounding_boxes)}: {bbox}")

                    location = bbox["location"]
                    extent = bbox["extent"]
                    rotation = bbox["rotation"]

                    # Compute 3D corners and project to 2D
                    corners_3d = self.compute_bounding_box_3d_lidar(location, extent, rotation, ego_location)
                    logging.debug(f"Computed 3D corners for bounding box {idx + 1}: {corners_3d}")

                    x_min, y_min, x_max, y_max = self.project_bbox_to_2d_lidar(corners_3d, frame_size, lidar_range)
                    logging.debug(f"Projected bounding box {idx + 1} to 2D: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                    # Validate coordinates
                    if x_min < 0 or y_min < 0 or x_max > frame_width or y_max > frame_height:
                        logging.warning(f"Bounding box {idx + 1} has coordinates out of frame bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}. Skipping.")
                        continue

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    logging.info(f"Bounding box {idx + 1} drawn on the frame.")
                except Exception as bbox_error:
                    logging.error(f"Error processing bounding box {idx + 1}: {bbox_error}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save the updated frame
            output_frame_path = os.path.join(output_dir, os.path.basename(frame_path))
            cv2.imwrite(output_frame_path, frame)
            logging.info(f"Frame with bounding boxes saved to: {output_frame_path}")

        except Exception as e:
            logging.error(f"Error plotting bounding boxes: {e}")