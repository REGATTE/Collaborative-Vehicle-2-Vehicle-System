import pygame
import numpy as np
import carla
import logging
import queue


class EgoVehicleVisualizer:
    def __init__(self, sensors):
        """
        Initializes the Pygame window for visualizing sensor data.
        :param sensors: List of sensor objects.
        """
        self.sensors = sensors
        self.width = 640  # Default width for each sensor window
        self.height = 360  # Default height for each sensor window
        self.surface_map = {sensor.type_id: None for sensor in sensors}
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Ego Vehicle Sensors Visualization")

    def update_window(self, sensor_id, image):
        """
        Updates the Pygame window with the given sensor image.
        :param sensor_id: ID of the sensor to update.
        :param image: NumPy array representing the sensor image.
        """
        img_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.surface_map[sensor_id] = img_surface

        self.window.fill((0, 0, 0))  # Clear the window
        if self.surface_map[sensor_id]:
            self.window.blit(self.surface_map[sensor_id], (0, 0))
        pygame.display.flip()

    def close(self):
        """
        Closes the Pygame window.
        """
        pygame.quit()


def project_lidar_to_2d(points, img_size):
    """
    Projects LIDAR points to a 2D plane for visualization.
    :param points: LIDAR points as a numpy array.
    :param img_size: Tuple (width, height) of the visualization image.
    :return: A numpy array with the LIDAR projection.
    """
    lidar_data = np.array([
        (int((p[0] + 50) * img_size[0] / 100), int((p[1] + 50) * img_size[1] / 100))
        for p in points if -50 <= p[0] <= 50 and -50 <= p[1] <= 50
    ])
    lidar_data = lidar_data[
        (lidar_data[:, 0] >= 0) & (lidar_data[:, 0] < img_size[0]) &
        (lidar_data[:, 1] >= 0) & (lidar_data[:, 1] < img_size[1])
    ]
    lidar_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    lidar_img[lidar_data[:, 1], lidar_data[:, 0]] = [255, 255, 255]
    return lidar_img

def visualize_ego_sensors(world, sensors):
    """
    Visualize sensor data for the ego vehicle and allow toggling between sensors with the Tab key.
    :param world: CARLA world object.
    :param sensors: List of sensor objects attached to the ego vehicle.
    """
    current_sensor_index = 0  # Start with the first sensor
    sensor_data_queue = queue.Queue()  # Shared queue for sensor data
    visualizer = EgoVehicleVisualizer(sensors)

    def sensor_callback(sensor):
        def callback(data):
            """
            Callback function to process sensor data and store it in a shared queue.
            :param data: Sensor data.
            """
            try:
                sensor_type = sensor.type_id
                if 'camera' in sensor_type.lower():
                    # Process camera image
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = array.reshape((data.height, data.width, 4))[:, :, :3]
                    sensor_data_queue.put((sensor_type, array))
                elif 'lidar' in sensor_type.lower():
                    # Process lidar data
                    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                    lidar_img = project_lidar_to_2d(points, (640, 360))
                    sensor_data_queue.put((sensor_type, lidar_img))
                elif 'gnss' in sensor_type.lower():
                    # GPS data logged but not visualized
                    coords = f"Lat: {data.latitude:.6f}\nLon: {data.longitude:.6f}\nAlt: {data.altitude:.2f}"
                    logging.info(f"GPS Data: {coords}")
                    sensor_data_queue.put((sensor_type, None))
                elif 'depth' in sensor_type.lower():
                    # Process depth data directly
                    max_depth = 100.0  # Maximum depth for normalization (in meters)
                    raw_depth = np.frombuffer(data.raw_data, dtype=np.float32).reshape((data.height, data.width))
                    normalized_depth = np.clip(raw_depth / max_depth * 255, 0, 255).astype(np.uint8)  # Normalize depth to [0, 255]

                    # Convert to 3-channel grayscale for visualization
                    depth_image = np.stack([normalized_depth] * 3, axis=-1)
                    surface = pygame.surfarray.make_surface(depth_image.swapaxes(0, 1))
            except Exception as e:
                logging.error(f"Error in sensor callback: {e}")
        return callback

    try:
        pygame.init()
        screen = pygame.display.set_mode((640, 360), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Ego Vehicle Sensors Visualization")

        # Start listening to the first sensor
        sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))
        logging.info(f"Listening to sensor: {sensors[current_sensor_index].type_id}")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                    # Stop current sensor
                    sensors[current_sensor_index].stop()
                    current_sensor_index = (current_sensor_index + 1) % len(sensors)
                    sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))
                    logging.info(f"Switched to sensor: {sensors[current_sensor_index].type_id}")

            # Update display with the latest sensor data
            while not sensor_data_queue.empty():
                sensor_type, data = sensor_data_queue.get()
                if data is not None:
                    visualizer.update_window(sensor_type, data)

    except Exception as e:
        logging.error(f"Error in visualization: {e}")
    finally:
        # Cleanup sensors
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception as e:
                logging.warning(f"Failed to cleanup sensor {sensor.id}: {e}")
        visualizer.close()
