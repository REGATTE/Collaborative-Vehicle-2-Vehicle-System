import logging
import os
import cv2
import time
import numpy as np

def project_lidar_to_2d(lidar_points, frame_size=(1920, 1080), lidar_range=400):
    """
    Projects LiDAR points onto a 2D image plane.

    :param lidar_points: NumPy array of LiDAR points in XYZI format (N, 4).
    :param frame_size: Tuple indicating the dimensions of the output frame (width, height).
    :param lidar_range: The maximum range of the LiDAR sensor.
    :return: A 2D numpy array representing the projected LiDAR data.
    """
    try:
        width, height = frame_size
        lidar_image = np.zeros((height, width), dtype=np.uint8)

        scale_x = width / lidar_range
        scale_y = height / lidar_range

        # Log point stats for debugging
        logging.debug(f"Projecting {lidar_points.shape[0]} LiDAR points to a frame of size {frame_size}.")

        for point in lidar_points:
            x, y, z, intensity = point
            if z < -2:  # Filter ground clutter
                continue

            px = int((x + lidar_range / 2) * scale_x)
            py = int((y + lidar_range / 2) * scale_y)

            # Ensure pixel coordinates are within bounds
            if 0 <= px < width and 0 <= py < height:
                pixel_value = int(min(255, max(0, intensity * 255)))
                lidar_image[py, px] = pixel_value

        logging.info(f"LiDAR frame created successfully with {lidar_points.shape[0]} points.")
        return lidar_image
    except Exception as e:
        logging.error(f"Error projecting LiDAR to 2D: {e}")
        return None


def save_lidar_frames(lidar_points, frame_size=(1920, 1080), output_dir="combined_lidar_frames"):
    if lidar_points is None or len(lidar_points) == 0:
        logging.warning("No combined LiDAR data to save.")
        return None

    try:
        lidar_image = project_lidar_to_2d(lidar_points, frame_size)
        if lidar_image is not None:
            os.makedirs(output_dir, exist_ok=True)
            frame_filename = f"frame_{int(time.time() * 1000)}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, lidar_image)
            logging.info(f"LiDAR frame saved: {frame_path}")
            return frame_filename
    except Exception as e:
        logging.error(f"Error saving LiDAR frame: {e}")
        return None
