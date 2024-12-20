import socket
import threading
import logging
import json
import numpy as np
import time
import os, sys, cv2
import logging

from utils.proximity_mapping import ProximityMapping
from utils.vehicle_mapping.vehicle_mapping import load_vehicle_mapping
from utils.compression import DataCompressor
from utils.bbox.bbox import BoundingBoxExtractor
from utils.bbox.det_box import LidarBoundingBoxDetector
from utils.save_frame import save_lidar_frames

frames_dir = "combined_lidar_frames"
bboxes_dir = "combined_bounding_boxes"
output_dir = "frames_with_bboxes"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EgoVehicleListener:
    def __init__(self, lidar_data_buffer, lidar_data_lock, host='127.0.0.1', port=65432, ego_vehicle=None, world=None, lidar_range=400):
        """
        Initializes the ego vehicle listener.
        :param lidar_data_buffer: Shared buffer for storing LIDAR data.
        :param lidar_data_lock: Lock for synchronizing access to the LIDAR data buffer.
        :param host: Host address for the listener.
        :param port: Port for the listener.
        :param ego_vehicle: Ego vehicle actor.
        :param world: CARLA world instance.
        """
        if world is None:
            raise ValueError("CARLA world instance is required.")
        self.host = host
        self.port = port
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.lidar_range = lidar_range

        # shared data
        self.lidar_data_lock = lidar_data_lock  # Shared lock for thread safety
        self.lidar_data_buffer = lidar_data_buffer  # Shared buffer for LIDAR data

        # utilities
        self.vehicle_mapping = load_vehicle_mapping()  # Load vehicle mapping
        self.proximity_mapping = ProximityMapping(world=self.world, radius=20.0)  # Use proximity mapping
        self.bounding_box_extractor = BoundingBoxExtractor(
            world=self.world,
            ego_vehicle=self.ego_vehicle,
            vehicle_mapping=self.vehicle_mapping,
            proximity_mapping=self.proximity_mapping
        )

        # Runtime variables
        self.lidar_data_proximity = {}  # Store LIDAR data for vehicles in proximity
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
        Computes the pose of the smart vehicle relative to the ego vehicle using global positions.
        Ensures the smart vehicle's position is relative to the ego vehicle's position.

        :param vehicle_label: Label of the smart vehicle.
        :return: Dictionary with relative position and rotation.
        """
        vehicle_data = self.vehicle_mapping.get(vehicle_label)
        if not vehicle_data:
            raise ValueError(f"Vehicle label {vehicle_label} not found in mapping.")

        # Fetch the actor ID
        actor_id = vehicle_data.get("actor_id")
        if actor_id is None:
            logger.error(f"Actor ID not found for vehicle {vehicle_label}")
            raise ValueError(f"Actor ID is missing for vehicle {vehicle_label}")

        # Get the smart vehicle actor and its current transform
        smart_vehicle = self.world.get_actor(actor_id)
        if smart_vehicle is None:
            logger.error(f"Actor with ID {actor_id} not found in the simulation for {vehicle_label}")
            raise ValueError(f"Smart vehicle with actor ID {actor_id} not found")

        smart_transform = smart_vehicle.get_transform()
        smart_position = np.array([
            smart_transform.location.x,
            smart_transform.location.y,
            smart_transform.location.z
        ])
        smart_yaw = smart_transform.rotation.yaw

        # Get the global position of the ego vehicle
        ego_transform = self.ego_vehicle.get_transform()
        ego_position = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.location.z
        ])
        ego_yaw = ego_transform.rotation.yaw

        # Compute delta position (global position difference)
        delta_position = smart_position - ego_position

        # Convert ego yaw to radians
        ego_yaw_rad = np.radians(ego_yaw)

        # Rotation matrix for ego vehicle's orientation
        ego_rotation_matrix = np.array([
            [np.cos(ego_yaw_rad), np.sin(ego_yaw_rad), 0],
            [-np.sin(ego_yaw_rad), np.cos(ego_yaw_rad), 0],
            [0, 0, 1]
        ])

        # Transform the delta position into the ego vehicle's coordinate frame
        relative_position = np.dot(ego_rotation_matrix, delta_position)

        # Compute relative yaw
        relative_yaw = smart_yaw - ego_yaw

        # normalize realtive yaw to [-180, 180]
        relative_yaw = (relative_yaw + 180) % 360 - 180


        # Log details for debugging
        logger.debug(f"Global Smart Vehicle Position: {smart_position}")
        logger.debug(f"Global Ego Vehicle Position: {ego_position}")
        logger.debug(f"Delta Position (Global): {delta_position}")
        logger.debug(f"Relative Position (Ego Frame): {relative_position}")
        logger.debug(f"Smart Vehicle Yaw: {smart_yaw}°, Ego Vehicle Yaw: {ego_yaw}°")
        logger.debug(f"Relative Yaw: {relative_yaw}°")

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
        Transforms LIDAR points (in XYZI format) from a smart vehicle to the ego vehicle's coordinate frame.

        :param lidar_points: NumPy array of LIDAR points in XYZI format.
        :param relative_position: Dictionary with keys 'x', 'y', 'z' indicating the translation vector.
        :param relative_yaw: Relative yaw angle in degrees.
        :return: Transformed LIDAR points in XYZI format.
        """
        # Validate inputs
        if lidar_points.ndim != 2 or lidar_points.shape[1] != 4:
            raise ValueError("LIDAR points must be a 2D array with 4 columns (XYZI format).")

        relative_yaw_rad = np.radians(relative_yaw)

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(relative_yaw_rad), -np.sin(relative_yaw_rad), 0],
            [np.sin(relative_yaw_rad),  np.cos(relative_yaw_rad), 0],
            [0, 0, 1]
        ])

        # Convert lidar points to NumPy array
        lidar_points = np.array(lidar_points)

        # Split LIDAR data into XYZ and I (Intensity)
        xyz_points = lidar_points[:, :3]  # Extract X, Y, Z
        intensity = lidar_points[:, 3]    # Extract Intensity (I)

        logger.debug(f"First 5 raw LIDAR points:\n{lidar_points[:5]}")

        # Apply rotation matrix to all points
        rotated_points = np.dot(xyz_points, rotation_matrix.T)
        logger.debug(f"First 5 rotated points:\n{rotated_points[:5]}")

        # Add translation (relative position)
        translated_points = rotated_points + np.array([
            relative_position['x'],
            relative_position['y'],
            relative_position['z']
        ])

        return np.hstack((translated_points, intensity.reshape(-1, 1)))

    def combine_lidar_data(self, ego_vehicle):
        """
        Combines LIDAR data from all nearby smart vehicles into a single dataset and extracts bounding boxes.
        """
        combined_lidar = None
        gt_bounding_boxes = []
        det_bounding_boxes = []

        ego_location = {
            "x": self.ego_vehicle.get_transform().location.x,
            "y": self.ego_vehicle.get_transform().location.y,
            "z": self.ego_vehicle.get_transform().location.z
        }
        ego_yaw = self.ego_vehicle.get_transform().rotation.yaw

        # Access ego vehicle lidar data
        ego_lidar_id = 32  # Force mapping for testing
        logging.info(f"combine_lidar_data: Attempting to access data for Sensor ID {ego_lidar_id}.")

        with self.lidar_data_lock:
            ego_lidar_data = self.lidar_data_buffer.get(ego_lidar_id, [])
        # Convert to list from memoryview
        if isinstance(ego_lidar_data, memoryview):
            logging.info("Converting ego lidar data to list")
            ego_lidar_data = list(ego_lidar_data)

        if not ego_lidar_data:
            logging.warning(f"No LIDAR data found for Sensor ID {ego_lidar_id}.")
        else:
            try:
                # Process and add ego lidar data
                ego_lidar_array = np.frombuffer(bytearray(ego_lidar_data), dtype=np.float32).reshape(-1, 4)
                combined_lidar = ego_lidar_array
                logging.info(
                    f"Ego LIDAR data processed with {combined_lidar.shape[0]} points."
                )
                # Extract bounding boxes for the ego vehicle
                ego_bounding_boxes = self.bounding_box_extractor.extract_bounding_boxes(self.ego_vehicle, ego_location)
                if ego_bounding_boxes:
                    gt_bounding_boxes.extend(ego_bounding_boxes)
                    logging.info(f"Ego vehicle bounding boxes extracted: {len(ego_bounding_boxes)}.")
                else:
                    logging.warning("No bounding boxes extracted for the ego vehicle.")
            except Exception as e:
                logging.error(f"Error processing ego LIDAR data: {e}")

        # Process smart vehicles' LIDAR data
        for vehicle_label, lidar_points in self.lidar_data_proximity.items():
            if vehicle_label not in self.vehicle_mapping:
                logging.warning(f"Vehicle label {vehicle_label} not found in mapping. Skipping.")
                continue
            try:
                # Convert list to numpy array for reshaping and manipulation
                smart_lidar_array = np.frombuffer(bytearray(lidar_points), dtype=np.float32).reshape(-1, 4)

                # Compute relative pose
                relative_pose = self.compute_relative_pose(vehicle_label)
                relative_position = relative_pose["relative_position"]
                relative_yaw = relative_pose["relative_yaw"]

                # Transform lidar points
                transformed_points = self.transform_lidar_points(smart_lidar_array, relative_position, relative_yaw)
                combined_lidar = np.vstack((combined_lidar, transformed_points)) if combined_lidar is not None else transformed_points

                # Extract bounding boxes for the smart vehicle
                vehicle_actor = self.proximity_mapping.world.get_actor(self.vehicle_mapping[vehicle_label]["actor_id"])
                if vehicle_actor:
                    vehicle_bounding_boxes = self.bounding_box_extractor.extract_bounding_boxes(vehicle_actor, ego_location)
                    if vehicle_bounding_boxes:
                        gt_bounding_boxes.extend(vehicle_bounding_boxes)
                        logging.info(f"Bounding boxes extracted for vehicle {vehicle_label}: {len(vehicle_bounding_boxes)}.")
                    else:
                        logging.warning(f"No bounding boxes extracted for vehicle {vehicle_label}.")
                else:
                    logging.warning(f"Vehicle actor not found for {vehicle_label}.")
            except Exception as e:
                logging.error(f"Error processing LIDAR data for vehicle {vehicle_label}: {e}")

        # Save the combined LiDAR frame
        frame_file = None
        if combined_lidar is not None:
            frame_file = save_lidar_frames(
                lidar_points=combined_lidar,
                frame_size=(1920, 1080),
                output_dir="frames/combined_lidar_frames"
            )

        # Save gt bounding boxes separately
        gt_bbox_file = None
        if gt_bounding_boxes:
            gt_bbox_file = self.bounding_box_extractor.save_bounding_boxes(
                bounding_boxes=gt_bounding_boxes,
                output_dir="frames/gt_bounding_boxes"
            )
            if gt_bbox_file:
                logging.info(f"GT Bounding boxes saved: {gt_bbox_file}.")
            else:
                logging.warning("GT Bounding boxes were not saved correctly. File name is None.")
        else:
            logging.warning("No gt bounding boxes extracted.")

        # Save the frame with gt bounding boxes plotted
        if frame_file and gt_bbox_file:
            logging.debug(
                "Both frame_file and bbox_file are available. Proceeding to plot bounding boxes."
            )
            frame_path = os.path.join("frames/combined_lidar_frames", frame_file)
            gt_bbox_path = os.path.join("frames/gt_bounding_boxes", gt_bbox_file)
            output_dir = "frames/frames_with_gt_bboxes"

            logging.debug(f"Frame path: {frame_path}")
            logging.debug(f"Bounding boxes path: {gt_bbox_path}")
            logging.debug(f"Output directory for frames with bounding boxes: {output_dir}")

            try:
                # Plot bounding boxes on the frame
                self.bounding_box_extractor.plot_bounding_boxes_on_lidar_frame(
                    frame_path=frame_path,
                    bounding_boxes=gt_bounding_boxes,
                    frame_size=(1920, 1080),
                    lidar_range=400,  # Example lidar range
                    ego_location=ego_location,
                    output_dir=output_dir
                )
                logging.info(f"Frame with gt bounding boxes saved successfully to: {output_dir}")
            except Exception as e:
                logging.error(f"Failed to plot gt bounding boxes on frame. Error: {e}")
        else:
            if not frame_file:
                logging.warning("Frame file is missing. Skipping frame with bounding boxes plotting.")
            if not gt_bbox_file:
                logging.warning("GT Bounding boxes file is missing. Skipping frame with bounding boxes plotting.")

        logging.info(f"Combined LIDAR Data: {len(combined_lidar)} points across all nearby vehicles.")
        
        # Initialize the Lidar Bounding Box Detector
        lidar_det_bounding_box_detector = LidarBoundingBoxDetector(output_dir="frames/det_bounding_boxes")
        
        # Detect bounding boxes
        try:
            det_bounding_boxes = lidar_det_bounding_box_detector.detect_and_transform_bounding_boxes(
                point_cloud=combined_lidar,
                ego_location=ego_location,
                ego_yaw=ego_yaw,
                eps=0.5,
                min_samples=10
            )
            logging.info(f"Detected {len(det_bounding_boxes)} bounding boxes.")
        except Exception as e:
            logging.error(f"Error during bounding box detection: {e}")
            det_bounding_boxes = []
        
        # Save detected bounding boxes
        det_bbox_file = None
        if det_bounding_boxes:
            try:
                det_bbox_file = lidar_det_bounding_box_detector.save_bounding_boxes(
                    bounding_boxes=det_bounding_boxes,
                    output_dir="frames/det_bounding_boxes"
                )
                if det_bbox_file:
                    logging.info(f"Detected bounding boxes saved to file: {det_bbox_file}.")
                else:
                    logging.warning("Detected bounding boxes were not saved correctly. File name is None.")
            except Exception as e:
                logging.error(f"Failed to save detected bounding boxes. Error: {e}")
        else:
            logging.warning("No detected bounding boxes extracted to save.")

        # Plot detected bounding boxes on the ground truth LiDAR frame
        if frame_file and det_bounding_boxes:
            logging.debug("Both frame_file and detected bounding boxes are available. Proceeding to plot bounding boxes.")
            frame_path = os.path.join("frames/frames_with_gt_bboxes", frame_file)
            output_dir = "frames/frames_with_det_bboxes"

            logging.debug(f"Frame path: {frame_path}")
            logging.debug(f"Output directory for frames with detected bounding boxes: {output_dir}")

            try:
                # Plot detected bounding boxes on the frame
                logging.info("Plotting bounding boxes on the LiDAR frame...")
                lidar_det_bounding_box_detector.plot_bounding_boxes_on_lidar_frame(
                    frame_path=frame_path,
                    bounding_boxes=det_bounding_boxes,
                    frame_size=(1920, 1080),
                    lidar_range=400,  # Example lidar range
                    ego_location=ego_location,
                    output_dir=output_dir
                )
                logging.info(f"Frame with detected bounding boxes saved successfully to: {output_dir}")
            except Exception as e:
                logging.error(f"Failed to plot detected bounding boxes on frame. Error: {e}")
        else:
            if not frame_file:
                logging.warning("Frame file is missing. Skipping frame with bounding boxes plotting.")
            if not det_bounding_boxes:
                logging.warning("Detected bounding boxes are missing. Skipping frame with bounding boxes plotting.")

        logging.info(f"Combined LIDAR Data: {len(combined_lidar)} points processed successfully.")

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
            try:
                conn, addr = server_socket.accept()
                logging.info(f"Connection accepted from {addr}. Starting thread for handle_connection.")
                threading.Thread(target=self.handle_connection, args=(conn, addr), daemon=True).start()
            except Exception as e:
                logging.error(f"Error accepting connection: {e}")
        server_socket.close()
        logging.info("Listener stopped")

    def handle_connection(self, conn, addr):
        """
        Handles incoming connections from smart vehicles and ensures continuous processing of data.
        """
        logging.info(f"handle_connection started for {addr}.")
        data_decompressor = DataCompressor()
        try:
            with conn:
                buffer = b""
                while self.running:
                    try:
                        chunk = conn.recv(49152) # accpets 48Kb
                        if not chunk:
                            if buffer:
                                logging.warning(f"Incomplete data received from {addr}. Closing connection.")
                            else:
                                logging.warning(f"No data received from {addr}. Closing connection.")
                            break

                        buffer += chunk  # Accumulate chunks
                        try:
                            # Parse the received JSON data
                            smart_data = json.loads(buffer.decode('utf-8'))
                            buffer = b""
                        except json.JSONDecodeError:
                            # If JSON is incomplete, wait for more data
                            continue                        
                        smart_vehicle_id = smart_data.get('id')

                        if not smart_vehicle_id:
                            logging.warning(f"Smart vehicle ID missing in data from {addr}.")
                            continue

                        vehicle_label = self.get_vehicle_label(smart_vehicle_id)

                        if not vehicle_label:
                            logging.warning(f"Smart Vehicle ID {smart_vehicle_id} not found in mapping.")
                            continue

                        #decompress lidar data
                        lidar_data = smart_data.get('lidar')
                        if lidar_data:
                            try:
                                decompressed_lidar_data = data_decompressor.decompress(lidar_data)
                                smart_data['lidar'] = decompressed_lidar_data
                                logging.info(f"LIDAR data decompressed successfully for Smart Vehicle {vehicle_label}.")
                            except Exception as e:
                                logging.error(f"Error decompressing LIDAR data for Smart Vehicle {vehicle_label}: {e}")
                                continue
                        else:
                            logging.warning(f"No LIDAR data found for Smart Vehicle {vehicle_label}.")
                        logging.info(f"Processing data from Smart Vehicle {vehicle_label} (ID: {smart_vehicle_id}).")

                        # Process data only if the vehicle is in proximity
                        self._process_data_if_in_proximity(vehicle_label, smart_data)

                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decoding error for data from {addr}: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error while processing data from {addr}: {e}")
        except Exception as e:
            logging.error(f"Error in handle_connection for {addr}: {e}")
        finally:
            logging.info(f"Connection with {addr} closed.")

    def _process_data_if_in_proximity(self, vehicle_label, smart_data):
        """
        Processes data only if the vehicle is within proximity of the ego vehicle.
        """
        logging.info(f"Checking proximity for Smart Vehicle {vehicle_label}.")
        actors = self._get_all_valid_actors()

        if not actors:
            logging.error("No valid actors found in the CARLA world.")
            return

        vehicles_in_radius = self.proximity_mapping.find_vehicles_in_radius(self.ego_vehicle, actors)
        if vehicles_in_radius:
            logging.info(f"Vehicles in radius: {vehicles_in_radius}")
        else:
            logging.info("No vehicles in proximity of the Ego Vehicle.")

        if smart_data['id'] in vehicles_in_radius:
            logging.info(f"Smart Vehicle {vehicle_label} is in proximity. Processing LIDAR and relative pose.")
            self._process_vehicle_data(vehicle_label, smart_data)
        else:
            logging.debug(f"Smart Vehicle {vehicle_label} is not in proximity. Ignoring data.")

    def _process_vehicle_data(self, vehicle_label, smart_data):
        """
        Processes data received from a smart vehicle, dynamically updating combined LiDAR data.
        """
        try:
            lidar_points = smart_data.get('lidar', [])
            if not lidar_points:
                logging.warning(f"No LIDAR data available in the received data for {vehicle_label}.")
                return

            logging.info(f"Received {len(lidar_points)} LIDAR points from {vehicle_label}.")
            self.lidar_data_proximity[vehicle_label] = lidar_points

            combined_lidar = self.combine_lidar_data(self.ego_vehicle)
            logging.info(f"Path planning can now use combined LIDAR data with {len(combined_lidar)} points.")

            self.trigger_path_planning(combined_lidar)
        except Exception as e:
            logging.error(f"Error processing data for {vehicle_label}: {e}")

    def trigger_path_planning(self, combined_lidar):
        """
        Placeholder for path planning logic using combined LiDAR data.
        """
        logging.info(f"Triggering path planning with {len(combined_lidar)} combined LIDAR points.")

    def _get_all_valid_actors(self):
        """
        Retrieves all valid actors from the vehicle mapping for proximity calculations.
        """
        actors = []
        for label, data in self.vehicle_mapping.items():
            actor_id = data.get("actor_id")
            if actor_id is None:
                continue

            actor = self.proximity_mapping.world.get_actor(actor_id)
            if actor is None:
                logging.warning(f"Actor with ID {actor_id} (label: {label}) not found in CARLA world.")
            else:
                logging.info(f"Actor with ID {actor_id} found: {actor.type_id}")
                # logging.debug(f"Actor Details: {actor.attributes}")  # Log actor attributes for additional details
                actors.append(actor)
        return actors

    def stop_listener(self):
        """
        Stops the listener gracefully.
        """
        self.running = False
