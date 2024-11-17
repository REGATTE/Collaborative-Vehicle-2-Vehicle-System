import pygame
import numpy as np
import carla
import logging
import threading
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
        self.surface_map = {}
        pygame.init()
        self.window = pygame.display.set_mode((self.width * len(sensors), self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Ego Vehicle Sensors Visualization")

    def update_window(self, sensor_id, image):
        """
        Updates the Pygame window with the given sensor image.
        :param sensor_id: ID of the sensor to update.
        :param image: NumPy array representing the sensor image.
        """
        if sensor_id in self.surface_map:
            img_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.surface_map[sensor_id] = img_surface

        self.window.fill((0, 0, 0))  # Clear the window
        for idx, (sensor_id, surface) in enumerate(self.surface_map.items()):
            self.window.blit(surface, (self.width * idx, 0))
        pygame.display.flip()

    def close(self):
        """
        Closes the Pygame window.
        """
        pygame.quit()


import queue
import threading

def visualize_ego_sensors(world, sensors):
    """
    Visualize sensor data for the ego vehicle and allow toggling between sensors with the Tab key.
    :param world: CARLA world object.
    :param sensors: List of sensor objects attached to the ego vehicle.
    """
    current_sensor_index = 0  # Start with the first sensor
    sensor_data_queue = queue.Queue()  # Shared queue for sensor data

    def sensor_callback(sensor):
        def callback(data):
            """
            Callback function to process sensor data and store it in a shared queue.
            :param data: Sensor data.
            """
            sensor_type = sensor.type_id
            if 'camera' in sensor_type.lower():
                # Process camera image
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((data.height, data.width, 4))[:, :, :3]
                sensor_data_queue.put((sensor_type, array))
            elif 'lidar' in sensor_type.lower():
                # Process lidar data
                lidar_image = np.zeros((360, 640, 3), dtype=np.uint8)  # Example size
                sensor_data_queue.put((sensor_type, lidar_image))
            elif 'gnss' in sensor_type.lower():
                # Log GPS data as text
                coords = f"Lat: {data.latitude:.6f}, Lon: {data.longitude:.6f}, Alt: {data.altitude:.2f}"
                logging.info(f"GPS Data: {coords}")
                sensor_data_queue.put((sensor_type, None))
        return callback

    # Attach callbacks to sensors
    sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))

    try:
        pygame.init()
        window_width, window_height = 640, 360  # Default visualization size
        screen = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Ego Vehicle Sensors Visualization")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                    # Stop current sensor and switch to the next
                    sensors[current_sensor_index].stop()
                    current_sensor_index = (current_sensor_index + 1) % len(sensors)
                    sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))

            # Update display with the latest sensor data
            while not sensor_data_queue.empty():
                sensor_type, data = sensor_data_queue.get()
                if data is not None:
                    surface = pygame.surfarray.make_surface(data.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
        pygame.quit()
