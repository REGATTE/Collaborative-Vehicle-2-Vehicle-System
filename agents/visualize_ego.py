import pygame
import logging
import carla
import json
import os, sys
import numpy as np
from queue import Queue

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vehicle_mapping.vehicle_mapping import load_vehicle_mapping
from Simulation.sensors import Sensors
from utils.logging_config import configure_logging


# Visualization helper class
class EgoVehicleVisualizer:
    """
    Visualizer for displaying sensor data from the ego vehicle.
    """

    def __init__(self, sensors):
        """
        Initialize the visualizer with the given sensors.
        :param sensors: List of sensors attached to the ego vehicle.
        """
        self.sensors = sensors
        self.window = None

    def initialize_window(self):
        """
        Initialize the PyGame window for visualization.
        """
        pygame.init()
        self.window = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Ego Vehicle Sensor Visualization")

    def update_window(self, sensor_type, data):
        """
        Update the visualization window with sensor data.
        :param sensor_type: Type of the sensor (e.g., 'camera', 'lidar').
        :param data: Sensor data to visualize.
        """
        if self.window is None:
            self.initialize_window()

        if "camera" in sensor_type.lower():
            surface = pygame.surfarray.make_surface(np.flip(data, axis=1))
            self.window.blit(surface, (0, 0))
        elif "lidar" in sensor_type.lower():
            surface = pygame.surfarray.make_surface(data)
            self.window.blit(surface, (0, 0))
        
        pygame.display.flip()

    def close(self):
        """
        Close the visualizer and clean up resources.
        """
        pygame.quit()
     
# Main function for visualizing ego vehicle sensors
def visualize_ego_sensors(ego_vehicle_mapping):
    """
    Visualizes sensor data for the designated ego vehicle.
    :param ego_vehicle_mapping: Mapping containing the ego vehicle and its sensors.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Load ego vehicle and sensors
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
                    lidar_img = np.zeros((720, 1280), dtype=np.uint8)  # Placeholder for LIDAR visualization
                    sensor_data_queue.put((sensor_type, lidar_img))
                elif "gnss" in sensor_type.lower():
                    coords = f"Lat: {data.latitude:.6f}, Lon: {data.longitude:.6f}, Alt: {data.altitude:.2f}"
                    logging.info(f"GNSS Data: {coords}")
            except Exception as e:
                logging.error(f"Error in sensor callback: {e}")
        return callback

    try:
        for sensor in sensors:
            sensor.listen(sensor_callback(sensor))
            logging.info(f"Listening to sensor: {sensor.type_id}")

        visualizer.initialize_window()
        running = True
        current_sensor_index = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP and event.key == pygame.K_TAB:
                    sensors[current_sensor_index].stop()
                    current_sensor_index = (current_sensor_index + 1) % len(sensors)
                    sensors[current_sensor_index].listen(sensor_callback(sensors[current_sensor_index]))
                    logging.info(f"Switched to sensor: {sensors[current_sensor_index].type_id}")

            while not sensor_data_queue.empty():
                sensor_type, data = sensor_data_queue.get()
                if data is not None:
                    visualizer.update_window(sensor_type, data)

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception as e:
                logging.warning(f"Failed to stop sensor {sensor.id}: {e}")
        visualizer.close()

if __name__ == "__main__":
    configure_logging()
    logging.info("Launching Ego Vehicle Visualizer...")

    # Load the vehicle mapping from JSON
    ego_vehicle_mapping = load_vehicle_mapping("../utils/vehicle_mapping/ego_vehicle_mapping.json")
    if ego_vehicle_mapping and "ego_veh" in ego_vehicle_mapping:
        visualize_ego_sensors(ego_vehicle_mapping)
    else:
        logging.error("Ego vehicle mapping not found or invalid.")
