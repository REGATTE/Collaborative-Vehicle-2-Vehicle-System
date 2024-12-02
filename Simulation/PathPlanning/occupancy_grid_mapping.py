import numpy as np
import logging
import cv2
import os
import time

class OccupancyGridMap:
    def __init__(self, grid_size, cell_size, world_size):
        """
        Initialise the Occupancy Grid map

        :param grid_size: Tuple (width, height) representing the grid dimensions.
        :param cell_size: Size of each grid cell in meters.
        :param world_size: Tuple (width, height) representing the physical size of the world.
        """
        logging.info("Initialised occupancy grid mapping")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.world_size = world_size
        self.grid = np.zeros(grid_size, dtype=int)  # Initialize grid with all free cells (0)
    
    def _world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.

        :param x: World x-coordinate.
        :param y: World y-coordinate.
        :return: Grid coordinates (i, j).
        """
        i = int((x + self.world_size[0] / 2) / self.cell_size)
        j = int((y + self.world_size[1] / 2) / self.cell_size)
        logging.debug(f"Converted world coordinates ({x}, {y}) to grid coordinates ({i}, {j}).")
        return i, j
    
    def camera_to_world_coordinates(self, camera_coords, ego_vehicle_location):
        """
        Transform 3D camera coordinates to world coordinates.

        :param camera_coords: Nx3 array of 3D points in camera coordinates.
        :param ego_vehicle_location: Ego vehicle's current world coordinates (x, y, z).
        :return: Nx3 array of 3D world coordinates.
        """
        try:
            # Ensure input is 2D
            if camera_coords.ndim != 1 and camera_coords.shape[-1] != 3:
                logging.error(f"Invalid camera coordinates shape: {camera_coords.shape}")
                raise ValueError("Camera coordinates must have 3 values per point.")
            if not np.isfinite(camera_coords).all():
                logging.error(f"Non-finite camera coordinates: {camera_coords}")
                raise ValueError("Camera coordinates contain invalid values.")
            logging.debug(f"Camera coordinates (reshaped if needed): {camera_coords}")

            # Extrinsic matrix (camera-to-world transformation)
            extrinsic_matrix = np.array([
                [1, 0, 0, 0.7],
                [0, 1, 0, 0.0],
                [0, 0, 1, 1.6]
            ])

            # Transform camera coordinates to world coordinates
            camera_coords_homogeneous = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))  # Convert to homogeneous
            logging.debug(f"Camera coordinates (homogeneous): {camera_coords_homogeneous}")
            
            world_coords = (extrinsic_matrix @ camera_coords_homogeneous.T).T  # Matrix multiplication and transpose
            world_coords = world_coords[:, :3]  # Convert back to 3D coordinates

            # Add ego vehicle's location in the world
            world_coords += np.array([ego_vehicle_location["x"], ego_vehicle_location["y"], ego_vehicle_location["z"]])
            logging.debug(f"Transformed world coordinates: {world_coords}")
            return world_coords
        except Exception as e:
            logging.error(f"Error in camera_to_world_coordinates: {e}")
            raise

    
    def project_lidar_to_image(self, points, camera_intrinsics):
        """
        Project 3D LiDAR points to the image plane using the intrinsic camera matrix.

        :param points: Nx3 array of LiDAR points in camera coordinates.
        :param camera_intrinsics: Intrinsic camera parameters as a dictionary.
        :return: Nx2 array of 2D image coordinates.
        """
        fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
        cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]

        # Extract coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Log raw points
        logging.info(f"LiDAR points (camera frame): {points}")

        # Filter out points with z <= 0
        valid = z > 0
        x, y, z = x[valid], y[valid], z[valid]

        # Project to 2D
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        projected_points = np.stack((u, v), axis=1)

        # Log projected points
        logging.info(f"Projected points on the image plane: {projected_points}")

        return projected_points

    def estimate_depth_from_lidar(self, image_x, image_y, lidar_data, camera_intrinsics):
        """
        Estimate depth at the given image coordinates using LiDAR data.

        :param image_x: Image x-coordinate.
        :param image_y: Image y-coordinate.
        :param lidar_data: LiDAR point cloud data.
        :param camera_intrinsics: Camera intrinsic matrix.
        :return: Depth (z-coordinate in camera space).
        """

        try:
            # Extract spatial coordinates and intensity
            lidar_points_camera = lidar_data[:, :3]  # Extract x, y, z
            intensities = lidar_data[:, 3]  # Extract intensity

            # Filter points with low intensity (adjust threshold as needed)
            intensity_threshold = 0.5
            valid_intensity = intensities > intensity_threshold
            lidar_points_camera = lidar_points_camera[valid_intensity]

            # Filter points where z > 0 (points in front of the camera)
            valid_points = lidar_points_camera[:, 2] > 0
            lidar_points_camera = lidar_points_camera[valid_points]

            if lidar_points_camera.shape[0] == 0:
                logging.warning("No valid LiDAR points found.")
                return None
            
            # Extract intrinsic parameters
            fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
            cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]

            # Project LiDAR points to the image plane
            x_camera, y_camera, z_camera = lidar_points_camera[:, 0], lidar_points_camera[:, 1], lidar_points_camera[:, 2]
            u = (x_camera * fx / z_camera) + cx
            v = (y_camera * fy / z_camera) + cy
            projected_points = np.stack((u, v), axis=1)

            # Log projected points for debugging
            logging.debug(f"Projected LiDAR points (first 5): {projected_points[:5]}")  # Log first 5 points
            logging.debug(f"Corresponding depths (first 5): {z_camera[:5]}")  # Log first 5 depths

            # Calculate distances to the image point
            distances = np.sqrt((projected_points[:, 0] - image_x) ** 2 +
                                (projected_points[:, 1] - image_y) ** 2)

            # Find the closest projected point to the given image coordinates
            min_idx = np.argmin(distances)
            closest_distance = distances[min_idx]
            closest_depth = z_camera[min_idx]

            # Log details of the closest point
            logging.info(f"Closest point index: {min_idx}, distance: {closest_distance:.2f}, depth: {closest_depth:.2f}")

            # Dynamic threshold adjustment based on data statistics
            dynamic_threshold = min(5000, max(closest_distance * 2, 100))  # Example: threshold depends on closest distance
            logging.info(f"Dynamic threshold: {dynamic_threshold:.2f}")

            # Check if the closest point is within the threshold
            if closest_distance < dynamic_threshold:
                return closest_depth
            else:
                logging.warning(f"No valid depth found for image point ({image_x}, {image_y}). Closest distance: {closest_distance:.2f}")
                return None

        except Exception as e:
            logging.error(f"Error estimating depth: {e}")
            return None

    def update_with_obstacles(self, detections, ego_vehicle_location, camera_intrinsics, lidar_data):
        """
        Update the occupancy grid based on detected obstacles.

        :param detections: List of detected obstacles in image coordinates [(x, y), ...].
        :param ego_vehicle_location: Ego vehicle's current world coordinates (x, y, z).
        :param camera_intrinsics: Intrinsic camera parameters as a dictionary (focal lengths, principal points).
        :param lidar_data: LiDAR point cloud data.
        """
        logging.info("Updating occupancy grid with detected obstacles...")

        # Validate detections
        # Validate detections
        if not isinstance(detections, list) or not all(isinstance(det, (list, tuple)) and len(det) == 2 for det in detections):
            logging.error(f"Invalid detection format: {detections}. Expected a list of (x, y) tuples.")
            return

        updated_cells = 0

        for det in detections:
            try:
                # Image coordinates (bounding box center)
                image_x, image_y = det

                # Ensure indices are integers for any array indexing
                int_image_x = int(image_x)
                int_image_y = int(image_y)

                # Approximate depth using LiDAR data
                depth = self.estimate_depth_from_lidar(int_image_x, int_image_y, lidar_data, camera_intrinsics)
                if depth is None or depth <= 0:
                    logging.warning(f"Invalid or missing depth ({depth}) for detection at ({image_x}, {image_y}). Skipping...")
                    continue

                # Back-project to camera coordinates
                camera_coords = np.array([(image_x - camera_intrinsics["cx"]) * depth / camera_intrinsics["fx"],
                                        (image_y - camera_intrinsics["cy"]) * depth / camera_intrinsics["fy"],
                                        depth]).reshape(1, -1)

                # Transform to world coordinates
                world_coords = self.camera_to_world_coordinates(camera_coords, ego_vehicle_location)
                world_x, world_y, _ = world_coords.flatten()

                # Log details
                # Log transformed coordinates
                logging.info(f"Detection at ({image_x}, {image_y}): Depth={depth:.2f}, "
                            f"Camera Coords={camera_coords}, World Coords=({world_x:.2f}, {world_y:.2f})")
    
                # Convert world coordinates to grid indices
                grid_x, grid_y = self._world_to_grid(world_x, world_y)
                grid_x = max(0, min(self.grid_size[0] - 1, int(grid_x)))
                grid_y = max(0, min(self.grid_size[1] - 1, int(grid_y)))
                if not (0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]):
                    logging.warning(f"Grid indices ({grid_x}, {grid_y}) out of bounds for world coordinates ({world_x:.2f}, {world_y:.2f}).")
                    continue

                # Mark the cell in the grid
                self.grid[grid_y, grid_x] = 1
                updated_cells += 1
            except Exception as e:
                logging.error(f"Error updating obstacle: {det}, {e}")

        logging.info(f"Updated {updated_cells} grid cells with obstacles.")

    def reset(self):
        """Reset the grid to all free cells."""
        logging.info("Resetting Occupancy Grid Map to all free cells.")
        self.grid.fill(0)

class OverlayGrid:
    def __init__(self, grid_map, save_folder="frames/OccMap"):
        """
        Initialize the overlay grid for visualization.

        :param grid_map: OccupancyGridMap object.
        :param save_folder: Folder where the grid images will be saved.
        """
        logging.info("Initializing Overlay Grid...")
        self.grid_map = grid_map
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def save_grid(self, frame_number, timestamp=None):
        """
        Save the occupancy grid as an image.

        :param frame_number: The frame number to use in the file name.
        :param timestamp: Optional timestamp to include in the file name.
        """
        grid = self.grid_map.grid
        grid_height, grid_width = grid.shape

        # Convert grid to an image
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                if grid[i, j] == 1:  # Occupied cell
                    grid_image[i, j] = (0, 0, 255)  # Red
                else:  # Free cell
                    grid_image[i, j] = (0, 255, 0)  # Green

        # Construct the file name with optional timestamp
        if timestamp is None:
            timestamp = int(time.time())  # Use current time if timestamp is not provided

        file_name = os.path.join(
            self.save_folder,
            f"grid_{frame_number:04d}_{timestamp}.png"
        )

        # Save the grid image
        cv2.imwrite(file_name, grid_image)
        logging.info(f"Saved grid frame {frame_number} to {file_name}.")
