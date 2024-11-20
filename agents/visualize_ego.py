import pygame
import logging
import carla
import json
import os
import sys
import numpy as np
from queue import Queue, Empty
import cv2

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vehicle_mapping.vehicle_mapping import load_vehicle_mapping
from Simulation.sensors import Sensors
from utils.logging_config import configure_logging


class EgoVehicleVisualizer:
    """
    Visualizer for displaying sensor data from the ego vehicle.
    """

    def __init__(self, sensors):
        self.sensors = sensors
        self.window = None

    def initialize_window(self):
        pygame.init()
        self.window = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Ego Vehicle Sensor Visualization")

    def update_window(self, sensor_type, data):
        if self.window is None:
            self.initialize_window()

        if "camera" in sensor_type.lower():
            surface = pygame.surfarray.make_surface(np.flip(data, axis=1))
            self.window.blit(surface, (0, 0))
        elif "lidar" in sensor_type.lower():
            surface = pygame.surfarray.make_surface(data)
            self.window.blit(surface, (0, 0))

        pygame.display.flip()

    def update_title(self, sensor_name, sensor_id):
        pygame.display.set_caption(f"Sensor: {sensor_name}, ID: {sensor_id}")

    def close(self):
        pygame.quit()


def visualize_ego_sensors(ego_vehicle_mapping):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    ego_vehicle_id = ego_vehicle_mapping["ego_veh"]["actor_id"]
    sensors_data = ego_vehicle_mapping["ego_veh"]["sensors"]
    ego_vehicle = world.get_actor(ego_vehicle_id)

    if not ego_vehicle:
        logging.error(f"Ego vehicle with ID {ego_vehicle_id} not found in the world.")
        return

    sensors = [world.get_actor(sensor_id) for sensor_id in sensors_data if sensor_id]
    if not sensors:
        logging.error(f"No sensors found for ego vehicle ID {ego_vehicle_id}.")
        return

    logging.info(f"Visualizing sensors for ego vehicle ID: {ego_vehicle_id}")

    sensor_data_queue = Queue()
    visualizer = EgoVehicleVisualizer(sensors)

    def sensor_callback(sensor):
        def callback(data):
            try:
                sensor_type = sensor.type_id
                if "camera" in sensor_type.lower():
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = array.reshape((data.height, data.width, 4))[:, :, :3]
                    array = np.rot90(array, -1)
                    sensor_data_queue.put((sensor_type, array))
                elif "lidar" in sensor_type.lower():
                    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                    lidar_img = project_lidar_to_2d(points, (1280, 720))
                    sensor_data_queue.put((sensor_type, lidar_img))
                elif "gnss" in sensor_type.lower():
                    coords = f"Lat: {data.latitude:.6f}, Lon: {data.longitude:.6f}, Alt: {data.altitude:.2f}"
                    logging.info(f"GNSS Data: {coords}")
            except Exception as e:
                logging.error(f"Error in sensor callback: {e}")
        return callback

    def project_lidar_to_2d(points, image_size):
        """
        Projects LiDAR points onto a 2D image plane with higher density.
        :param points: LiDAR point cloud (Nx4).
        :param image_size: Tuple of (width, height) for the 2D image.
        :return: 2D image with projected LiDAR points.
        """
        lidar_image = np.zeros(image_size, dtype=np.uint8)

        # Scale factor to increase resolution
        scale_factor = 2

        # Adjusting image size to account for scaling
        width, height = image_size
        scaled_width, scaled_height = width * scale_factor, height * scale_factor
        scaled_lidar_image = np.zeros((scaled_height, scaled_width), dtype=np.uint8)

        for point in points:
            x, y, z, intensity = point

            # Filter out points below a certain threshold
            if z > -2:  # Filter ground points
                px = int((x + 50) * scaled_width / 100)
                py = int((y + 50) * scaled_height / 100)
                if 0 <= px < scaled_width and 0 <= py < scaled_height:
                    # Higher density by accumulating intensity values
                    scaled_lidar_image[py, px] += min(255, int(255 * intensity))

        # Downscale to original size for smoother visualization
        lidar_image = cv2.resize(scaled_lidar_image, (width, height), interpolation=cv2.INTER_LINEAR)

        return lidar_image

    try:
        current_sensor_index = 0

        # Start listening to the first sensor
        sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))
        visualizer.update_title(
            sensors[current_sensor_index].type_id, sensors[current_sensor_index].id
        )
        logging.info(f"Listening to sensor: {sensors[current_sensor_index].type_id}")

        visualizer.initialize_window()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                    # Stop the current sensor
                    try:
                        sensors[current_sensor_index].stop()
                        logging.info(f"Stopped listening to sensor: {sensors[current_sensor_index].type_id}")
                    except Exception as e:
                        logging.warning(f"Failed to stop sensor: {e}")

                    # Switch to the next sensor
                    current_sensor_index = (current_sensor_index + 1) % len(sensors)

                    try:
                        sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))
                        visualizer.update_title(
                            sensors[current_sensor_index].type_id, sensors[current_sensor_index].id
                        )
                        logging.info(f"Switched to sensor: {sensors[current_sensor_index].type_id}")
                    except Exception as e:
                        logging.error(f"Failed to switch to sensor: {e}")

            while not sensor_data_queue.empty():
                try:
                    sensor_type, data = sensor_data_queue.get_nowait()
                    if data is not None:
                        visualizer.update_window(sensor_type, data)
                except Empty:
                    continue

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
                logging.info(f"Stopped sensor: {sensor.type_id}")
            except Exception as e:
                logging.warning(f"Failed to stop sensor {sensor.id}: {e}")
        visualizer.close()

if __name__ == "__main__":
    configure_logging()
    logging.info("Launching Ego Vehicle Visualizer...")

    try:
        ego_vehicle_mapping = load_vehicle_mapping("../utils/vehicle_mapping/vehicle_mapping.json")
        if ego_vehicle_mapping and "ego_veh" in ego_vehicle_mapping:
            visualize_ego_sensors(ego_vehicle_mapping)
        else:
            logging.error("Ego vehicle mapping not found or invalid.")
    except Exception as e:
        logging.error(f"Unhandled error in visualization: {e}")
