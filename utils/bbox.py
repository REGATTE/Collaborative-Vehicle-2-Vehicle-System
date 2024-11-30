import numpy as np
import logging
import os
import time
import json

class BoundingBoxExtractor:
    def __init__(self, world, ego_vehicle, vehicle_mapping, proximity_mapping):
        """
        Initializes the BoundingBoxExtractor

        :param ego_vehicle: The ego vehicle actor
        :param vehicle_mapping: Dictionary mapping vehicle labels to metadata
        :param proximity_mapping: Instance of proximityMapping to find vehicles nearby.
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.vehicle_mapping = vehicle_mapping
        self.proximity_mapping = proximity_mapping
    
    def extract_bounding_boxes(self, actor):
        """
        Extracts the bounding boxes for a given actor.
        :param actor: CARLA actor object.
        :return: List of bounding boxes for the actor.
        """
        try:
            bounding_boxes = []
            if actor:
                bounding_box = actor.bounding_box
                transform = actor.get_transform()
                bounding_boxes.append({
                    "location": {
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z
                    },
                    "extent": {
                        "x": bounding_box.extent.x,
                        "y": bounding_box.extent.y,
                        "z": bounding_box.extent.z
                    },
                    "rotation": {
                        "pitch": transform.rotation.pitch,
                        "yaw": transform.rotation.yaw,
                        "roll": transform.rotation.roll
                    }
                })
            return bounding_boxes
        except Exception as e:
            logging.error(f"Error extracting bounding boxes for actor: {e}")
            return []
    
    def extract_combined_bounding_boxes(self, lidar_data, bounding_boxes):
        """
        Filter and return bounding boxes that overlap with the combined LiDAR data.

        :param lidar_data: Combined LiDAR data (ego + smart vehicles) as a NumPy array.
        :param bounding_boxes: List of bounding boxes extracted from nearby vehicles.
        :return: Filtered list of bounding boxes overlapping with the LiDAR data.
        """
        filtered_bounding_boxes = []
        for bbox_data in bounding_boxes:
            label = bbox_data["label"]
            corners = bbox_data["corners"]

            # Check if any LiDAR points overlap with the bounding box
            in_bbox = self._check_lidar_overlap(lidar_data, corners)
            if in_bbox:
                filtered_bounding_boxes.append(bbox_data)
                logging.info(f"Bounding box for {label} overlaps with combined LiDAR data.")
            else:
                logging.debug(f"Bounding box for {label} does not overlap with LiDAR data.")

        return filtered_bounding_boxes
    
    def _check_lidar_overlap(self, lidar_data, bbox_corners):
        """
        Check if any LiDAR points fall within the bounding box.

        :param lidar_data: Combined LiDAR points (NumPy array).
        :param bbox_corners: Bounding box corners as a NumPy array.
        :return: Boolean indicating whether LiDAR data overlaps with the bounding box.
        """
        # Axis-aligned bounding box (AABB) check
        x_min, y_min, z_min = bbox_corners.min(axis=0)
        x_max, y_max, z_max = bbox_corners.max(axis=0)

        return (
            (lidar_data[:, 0] >= x_min)
            & (lidar_data[:, 0] <= x_max)
            & (lidar_data[:, 1] >= y_min)
            & (lidar_data[:, 1] <= y_max)
            & (lidar_data[:, 2] >= z_min)
            & (lidar_data[:, 2] <= z_max)
        ).any()
    
    def _get_valid_actors(self):
        """
        Retrieve valid actors in the CARLA world for bounding box extraction.

        :return: List of valid vehicle actors.
        """
        actors = []
        for label, data in self.vehicle_mapping.items():
            actor_id = data.get("actor_id")
            if actor_id is None:
                continue
            actor = self.proximity_mapping.world.get_actor(actor_id)
            if actor:
                actors.append(actor)
            else:
                logging.warning(f"Actor with ID {actor_id} not found in CARLA world.")
        return actors

    def _process_actor(self, actor):
        """
        Process an actor to extract its bounding box transformed to the ego vehicle's frame.

        :param actor: The vehicle actor.
        :return: Dictionary with label and transformed bounding box corners.
        """
        try:
            bbox = actor.bounding_box
            transform = actor.get_transform()
            bbox_corners = self._get_bounding_box_corners(bbox, transform)
            relative_pose = self._compute_relative_pose(actor)
            transformed_corners = self._transform_bounding_box(bbox_corners, relative_pose)

            return {
                "label": self._get_vehicle_label(actor.id),
                "corners": transformed_corners
            }
        except Exception as e:
            logging.error(f"Error processing actor {actor.id}: {e}")
            return None
        
    def _get_vehicle_label(self, actor_id):
        """
        Retrieve the vehicle label from the mapping using the actor ID.

        :param actor_id: ID of the vehicle actor.
        :return: Label of the vehicle.
        """
        return next(
            (label for label, data in self.vehicle_mapping.items() if data.get("actor_id") == actor_id),
            None
        )
    
    def _get_bounding_box_corners(self, bbox, transform):
        """
        Transform the bounding box corners to world coordinates.

        :param bbox: The bounding box object.
        :param transform: The transform of the vehicle.
        :return: NumPy array of world-space bounding box corners.
        """
        return np.array([[corner.x, corner.y, corner.z] for corner in bbox.get_world_vertices(transform)])
    
    def _compute_relative_pose(self, actor):
        """
        Compute the relative pose of the actor with respect to the ego vehicle.

        :param actor: The vehicle actor.
        :return: Dictionary containing relative position and yaw.
        """
        vehicle_transform = actor.get_transform()
        ego_transform = self.ego_vehicle.get_transform()

        delta_position = np.array([
            vehicle_transform.location.x - ego_transform.location.x,
            vehicle_transform.location.y - ego_transform.location.y,
            vehicle_transform.location.z - ego_transform.location.z
        ])

        ego_yaw_rad = np.radians(ego_transform.rotation.yaw)
        rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), np.sin(ego_yaw_rad), 0],
            [-np.sin(ego_yaw_rad), np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])

        relative_position = np.dot(rotation_matrix, delta_position)
        relative_yaw = vehicle_transform.rotation.yaw - ego_transform.rotation.yaw

        return {
            "relative_position": {
                "x": relative_position[0],
                "y": relative_position[1],
                "z": relative_position[2]
            },
            "relative_yaw": relative_yaw
        }

    def _transform_bounding_box(self, bbox_corners, relative_pose):
        """
        Transform bounding box corners to the ego vehicle's coordinate frame.

        :param bbox_corners: NumPy array of bounding box corners in world coordinates.
        :param relative_pose: Dictionary containing relative position and yaw.
        :return: NumPy array of transformed bounding box corners.
        """
        relative_yaw_rad = np.radians(relative_pose["relative_yaw"])
        rotation_matrix = np.array([
            [np.cos(relative_yaw_rad), -np.sin(relative_yaw_rad), 0],
            [np.sin(relative_yaw_rad),  np.cos(relative_yaw_rad), 0],
            [0, 0, 1]
        ])

        relative_position = np.array([
            relative_pose["relative_position"]["x"],
            relative_pose["relative_position"]["y"],
            relative_pose["relative_position"]["z"]
        ])

        return np.dot(rotation_matrix, (bbox_corners - relative_position).T).T
        
    def save_bounding_boxes(self, bounding_boxes, output_dir="bounding_boxes"):
        """
        Saves the extracted bounding boxes as a JSON file for debugging and visualization.

        :param bounding_boxes: List of bounding box data.
        :param output_dir: Directory where the bounding box files will be saved.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(output_dir, f"bounding_boxes_{timestamp}.json")

            with open(file_path, "w") as file:
                json.dump(bounding_boxes, file, indent=4)
            logging.info(f"Bounding boxes saved to {file_path}.")
        except Exception as e:
            logging.error(f"Error saving bounding boxes: {e}")
