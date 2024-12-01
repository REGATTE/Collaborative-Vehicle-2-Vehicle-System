import numpy as np
import logging
import cv2
import os

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

    def update_with_obstacles(self, obstacles):
        """
        Update the grid with detected obstacles.

        :param obstacles: List of obstacle positions [(x, y), ...].
        """
        logging.info("Updating occupancy grid with detected obstacles...")
        updated_cells = 0
        for obstacle in obstacles:
            grid_x, grid_y = self._world_to_grid(obstacle[0], obstacle[1])
            if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                self.grid[grid_x, grid_y] = 1  # Mark cell as occupied
                updated_cells += 1
        logging.info(f"Updated {updated_cells} grid cells with obstacles.")

    def reset(self):
        """Reset the grid to all free cells."""
        logging.info("Resetting Occupancy Grid Map to all free cells.")
        self.grid.fill(0)

    def visualize(self):
        """Visualize the occupancy grid as a simple ASCII map."""
        logging.info("Visualizing occupancy grid:")
        for row in self.grid:
            print("".join(["#" if cell == 1 else "." for cell in row]))

class OverlayGridOverlay:
    def __init__(self, grid_map, cell_size, world_size, save_folder="frames/OccMap"):
        """
        Initialize the overlay grid for visualization.
        :param grid_map: OccupancyGridMap object.
        :param cell_size: Size of each grid cell in meters.
        :param world_size: Tuple (width, height) representing world dimensions in meters.
        """
        logging.info("Initializing Overlay Grid...")
        self.grid_map = grid_map
        self.cell_size = cell_size
        self.world_size = world_size
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def create_overlay(self):
        """
        Create a visual representation of the occupancy grid map.
        :return: OpenCV image of the overlay.
        """
        logging.info("Creating overlay for Occupancy Grid Map...")
        grid = self.grid_map.grid
        grid_height, grid_width = grid.shape
        overlay = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i in range(grid_height):
            for j in range(grid_width):
                overlay[i, j] = (0, 0, 255) if grid[i, j] == 1 else (0, 255, 0)
        # Resize overlay to match the CARLA map/world dimensions
        logging.debug("Resizing overlay to match world dimensions.")
        overlay = cv2.resize(
            overlay,
            (int(self.world_size[0] / self.cell_size), int(self.world_size[1] / self.cell_size)),
            interpolation=cv2.INTER_NEAREST
        )
        return overlay

    def overlay_on_image(self, base_image):
        """
        Overlay the occupancy grid map on a base image.
        :param base_image: The image to overlay the grid on (e.g., CARLA camera view).
        :return: Combined image.
        """
        logging.info("Overlaying Occupancy Grid Map on the base image.")
        overlay = self.create_overlay()

        # Ensure both images have the same size
        overlay_resized = cv2.resize(overlay, (base_image.shape[1], base_image.shape[0]))

        # Add transparency to the overlay
        alpha = 0.5
        combined_image = cv2.addWeighted(base_image, 1.0, overlay_resized, alpha, 0)
        logging.debug("Overlay created successfully.")
        return combined_image
    
    def save_frame(self, combined_image, frame_number):
        """
        Save the overlaid image to a file.

        :param combined_image: The overlaid image to save.
        :param frame_number: The frame number to use in the file name.
        """
        file_name = os.path.join(self.save_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(file_name, combined_image)
        logging.info(f"Saved frame {frame_number} to {file_name}.")