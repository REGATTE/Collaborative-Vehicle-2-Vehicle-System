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
            py = int((lidar_range / 2 - y) * scale_y)

            # Ensure pixel coordinates are within bounds
            if 0 <= px < width and 0 <= py < height:
                pixel_value = int(min(255, max(0, intensity * 255)))
                lidar_image[py, px] = pixel_value

        logging.info(f"LiDAR frame created successfully with {lidar_points.shape[0]} points.")
        return lidar_image
    except Exception as e:
        logging.error(f"Error projecting LiDAR to 2D: {e}")
        return None


def save_lidar_frames(lidar_points, frame_size=(1920, 1080), output_dir="sensor_frames", lidar_range=400):
    """
    Saves the given LiDAR points as a 2D frame.

    :param lidar_points: NumPy array of LiDAR points in XYZI format.
    :param frame_size: Tuple indicating the dimensions of the output frame (width, height).
    :param output_dir: Directory where the frames will be saved.
    :param lidar_range: The maximum range of the LiDAR sensor.
    """
    if lidar_points is None or len(lidar_points) == 0:
        logging.warning("No LiDAR data to save.")
        return

    try:
        lidar_image = project_lidar_to_2d(lidar_points, frame_size=frame_size, lidar_range=lidar_range)
        if lidar_image is not None:
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"frame_{int(time.time() * 1000)}.png")
            cv2.imwrite(frame_path, lidar_image)
            logging.info(f"LiDAR frame saved: {frame_path}")
        else:
            logging.warning("LiDAR image was not generated successfully. Frame not saved.")
    except Exception as e:
        logging.error(f"Error saving LiDAR frame: {e}")