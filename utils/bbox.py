import numpy as np
import logging
import os
import time
import json
import cv2

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
    
    def extract_bounding_boxes(self, actor, ego_location, radius=35.0):
        """
        Extracts the bounding boxes for a given actor if it is within a certain radius of the ego vehicle.

        :param actor: CARLA actor object.
        :param ego_location: Dictionary with 'x', 'y', 'z' for the ego vehicle's location.
        :param radius: Maximum distance from the ego vehicle within which the bounding box is extracted.
        :return: List of bounding boxes for the actor.
        """
        try:
            if not actor:
                logging.warning("No actor provided for bounding box extraction.")
                return []

            # Retrieve the bounding box and transform
            bounding_box = actor.bounding_box
            transform = actor.get_transform()

            if not bounding_box or not transform:
                logging.warning(f"Actor {actor.id} does not have a valid bounding box or transform.")
                return []

            # Calculate the distance to the ego vehicle
            actor_location = transform.location
            distance = ((actor_location.x - ego_location['x']) ** 2 +
                        (actor_location.y - ego_location['y']) ** 2 +
                        (actor_location.z - ego_location['z']) ** 2) ** 0.5

            if distance > radius:
                logging.info(f"Actor ID={actor.id} is outside the radius of {radius}m. Skipping.")
                return []

            # Log the actor type for debugging
            logging.info(f"Extracting bounding box for Actor ID={actor.id}, Type={actor.type_id}, Distance={distance:.2f}m.")

            # Extract bounding box details
            bounding_boxes = [{
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
            }]

            # Log bounding box details
            logging.debug(f"Extracted bounding box: {bounding_boxes[0]} for Actor ID={actor.id}.")

            return bounding_boxes

        except AttributeError as e:
            logging.error(f"Actor {actor.id} is missing attributes: {e}")
            return []

        except Exception as e:
            logging.error(f"Error extracting bounding boxes for Actor ID={actor.id}: {e}")
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
        
    def save_bounding_boxes(self, bounding_boxes, output_dir="combined_bounding_boxes"):
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
            filename = f"bounding_boxes_{int(time.time() * 1000)}.json"
            filepath = os.path.join(output_dir, filename)

            # Save bounding boxes to the JSON file
            with open(filepath, "w") as f:
                json.dump(bounding_boxes, f, indent=4)

            logging.info(f"Bounding boxes saved to: {filepath}")
            return filename
        except Exception as e:
            logging.error(f"Failed to save bounding boxes: {e}")
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
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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


