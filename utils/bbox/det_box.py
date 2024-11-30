import numpy as np
import json
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt

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

    def save_bounding_boxes_to_json(self, bounding_boxes, file_name="bounding_boxes.json"):
        output_path = os.path.join(self.output_dir, file_name)
        with open(output_path, "w") as json_file:
            json.dump(bounding_boxes, json_file, indent=4)
        print(f"Bounding boxes saved to {output_path}")

    def plot_bounding_boxes_on_lidar_frame(self, lidar_points, bounding_boxes, ego_location, output_dir="frames_with_bboxes", frame_file="frame_with_bboxes.png"):
        """
        Visualize bounding boxes on the LiDAR frame, centered on the ego vehicle.

        :param lidar_points: Numpy array of shape (N, 3) or (N, 4) containing point cloud data.
        :param bounding_boxes: List of bounding box dictionaries.
        :param ego_location: Dictionary containing ego vehicle's location.
        :param output_dir: Directory to save the visualization.
        :param frame_file: Name of the output frame image file.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 12))
        plt.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.5, c="gray", label="LiDAR Points")

        for bbox in bounding_boxes:
            loc = bbox["location"]
            ext = bbox["extent"]
            rect = plt.Rectangle(
                (loc["x"] - ext["x"], loc["y"] - ext["y"]),
                2 * ext["x"],
                2 * ext["y"],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

        plt.scatter(ego_location["x"], ego_location["y"], c="blue", s=50, label="Ego Vehicle")
        plt.title("Bounding Boxes on LiDAR Frame")
        plt.xlabel("X-axis (m)")
        plt.ylabel("Y-axis (m)")
        plt.legend()
        output_path = os.path.join(output_dir, frame_file)
        plt.savefig(output_path)
        plt.close()
        print(f"Frame with bounding boxes saved to {output_path}")