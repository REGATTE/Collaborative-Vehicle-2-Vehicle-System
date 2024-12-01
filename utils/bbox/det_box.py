import numpy as np
import logging
import os
import json
import time
import cv2
from sklearn.cluster import DBSCAN

class LidarBoundingBoxDetector:
    def __init__(self, output_dir="frames/det_bounding_boxes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_and_transform_bounding_boxes(self, point_cloud, ego_location, ego_yaw, eps=0.5, min_samples=10):
        """
        Detect bounding boxes and transform them to the ego vehicle's frame.
        """
        try:
            # Detect bounding boxes
            detected_bounding_boxes = self._detect_bounding_boxes(point_cloud, eps, min_samples)

            return self._transform_bounding_boxes_to_ego_frame(
                detected_bounding_boxes, ego_location, ego_yaw
            )
        except Exception as e:
            logging.error(f"Error during bounding box detection and transformation: {e}")
            return []

    def _detect_bounding_boxes(self, point_cloud, eps=0.5, min_samples=10):
        """
        Detect bounding boxes from a point cloud using clustering.
        """
        try:
            # Use only x, y, z for clustering (ignore intensity or other fields)
            xyz_points = point_cloud[:, :3]
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_points)
            labels = clustering.labels_

            bounding_boxes = []
            for cluster_label in set(labels):
                if cluster_label == -1:  # Skip noise points
                    continue
                cluster_points = xyz_points[labels == cluster_label]
                bounding_boxes.append(self._fit_bounding_box(cluster_points))

            return bounding_boxes
        except Exception as e:
            logging.error(f"Error detecting bounding boxes: {e}")
            return []

    def _fit_bounding_box(self, cluster_points):
        """
        Fit an axis-aligned bounding box around the cluster points.
        """
        x_min, y_min, z_min = cluster_points.min(axis=0)
        x_max, y_max, z_max = cluster_points.max(axis=0)

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
            "rotation": {"yaw": 0.0},  # Assume no rotation for simplicity
        }

    def _transform_bounding_boxes_to_ego_frame(self, bounding_boxes, ego_location, ego_yaw):
        """
        Transform bounding boxes to the ego vehicle's coordinate frame.
        """
        ego_yaw_rad = np.radians(ego_yaw)
        rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), -np.sin(ego_yaw_rad), 0],
            [np.sin(ego_yaw_rad),  np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])

        transformed_bounding_boxes = []
        for bbox in bounding_boxes:
            bbox_location = np.array([bbox['location']['x'], bbox['location']['y'], bbox['location']['z']])
            relative_position = bbox_location - np.array([ego_location['x'], ego_location['y'], ego_location['z']])
            transformed_position = np.dot(rotation_matrix, relative_position)

            bbox['location'] = {
                "x": transformed_position[0],
                "y": transformed_position[1],
                "z": transformed_position[2],
            }
            transformed_bounding_boxes.append(bbox)

        return transformed_bounding_boxes

    def save_bounding_boxes(self, bounding_boxes, output_dir="frames/det_bounding_boxes"):
        """
        Save detected bounding boxes to a JSON file.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"det_bounding_boxes_{int(time.time() * 1000)}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(bounding_boxes, f, indent=4)

            logging.info(f"Bounding boxes saved to: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving bounding boxes: {e}")
            return None

    def plot_bounding_boxes_on_lidar_frame(self, frame_path, bounding_boxes, frame_size, lidar_range, ego_location, output_dir="frames_with_bboxes"):
        """
        Plot bounding boxes on a LiDAR frame and save the updated frame.
        """
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError("Frame could not be loaded.")
            
            for bbox in bounding_boxes:
                corners_3d = self._compute_bounding_box_corners(
                    bbox['location'], bbox['extent'], bbox['rotation'], ego_location
                )
                x_min, y_min, x_max, y_max = self._project_bbox_to_2d(corners_3d, frame_size, lidar_range)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            os.makedirs(output_dir, exist_ok=True)
            output_frame_path = os.path.join(output_dir, os.path.basename(frame_path))
            cv2.imwrite(output_frame_path, frame)
            logging.info(f"Updated frame saved with bounding boxes: {output_frame_path}")
        except Exception as e:
            logging.error(f"Error plotting bounding boxes on frame: {e}")

    def _compute_bounding_box_corners(self, location, extent, rotation, ego_location):
        """
        Compute 3D bounding box corners relative to the ego vehicle.
        """
        x, y, z = extent['x'], extent['y'], extent['z']
        corners = np.array([
            [x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z],
            [x, y, -z], [x, -y, -z], [-x, -y, -z], [-x, y, -z],
        ])
        yaw_rad = np.radians(rotation['yaw'])
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0, 0, 1],
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        location_vec = np.array([location['x'], location['y'], location['z']])
        world_corners = rotated_corners + location_vec
        ego_vec = np.array([ego_location['x'], ego_location['y'], ego_location['z']])
        return world_corners - ego_vec

    def _project_bbox_to_2d(self, corners_3d, frame_size, lidar_range):
        """
        Project 3D bounding box corners to 2D pixel coordinates.
        """
        width, height = frame_size
        scale_x = width / lidar_range
        scale_y = height / lidar_range

        corners_2d = []
        for corner in corners_3d:
            x, y, _ = corner
            px = int((x + lidar_range / 2) * scale_x)
            py = int((lidar_range / 2 - y) * scale_y)
            corners_2d.append((px, py))

        corners_2d = np.array(corners_2d)
        x_min, y_min = corners_2d.min(axis=0)
        x_max, y_max = corners_2d.max(axis=0)

        return int(x_min), int(y_min), int(x_max), int(y_max)
