import pygame
import numpy as np
import logging
from multiprocessing import Manager
from threading import Lock
import socket
import pickle

class BEVVisualizer:
    """
    Fetches and processes LiDAR data from the server.
    """

    def __init__(self, host='127.0.0.1', port=65433):
        self.host = host
        self.port = port
        self.socket = None

    def connect_to_server(self):
        """
        Establish a persistent connection to the LiDAR data server.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        logging.info(f"Connected to LiDAR Data Server at {self.host}:{self.port}")

    def fetch_lidar_data(self):
        """
        Fetch the latest LiDAR data buffer from the server over the persistent connection.
        :return: Dictionary containing LiDAR data from all sensors or an empty dictionary if no data is available.
        """
        try:
            data = b""
            while True:
                chunk = self.socket.recv(4096)
                if b"<END>" in chunk:
                    data += chunk.split(b"<END>")[0]
                    break
                data += chunk

            lidar_data_buffer = pickle.loads(data)  # Deserialize the data

            if not lidar_data_buffer:
                logging.warning("No LiDAR data received from server.")
                return {}

            logging.info(f"Fetched LiDAR data for {len(lidar_data_buffer.keys())} sensors.")
            return lidar_data_buffer

        except Exception as e:
            logging.error(f"Error fetching LiDAR data: {e}")
            return {}

    def close_connection(self):
        """
        Close the persistent connection to the LiDAR data server.
        """
        if self.socket:
            self.socket.close()
            logging.info("Disconnected from LiDAR Data Server.")

class BEVRenderer:
    """
    Renders LiDAR data in a Pygame window.
    """

    def __init__(self, window_size=(1280, 720)):
        self.window_size = window_size
        self.window = None

    def initialize_window(self):
        """
        Initializes the Pygame window.
        """
        pygame.init()
        pygame.font.init()  # Ensure fonts are initialized
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Bird's Eye View (BEV) Visualization - LiDAR")
        self.window.fill((0, 0, 0))
        pygame.display.flip()
        logging.info("Pygame window initialized successfully.")

    def draw_lidar_points(self, lidar_points):
        """
        Draws LiDAR points on the Pygame window.
        :param lidar_points: NumPy array of shape (N, 4) containing LiDAR points ([x, y, z, intensity]).
        """
        if self.window is None:
            self.initialize_window()

        self.window.fill((0, 0, 0))  # Clear the screen
        width, height = self.window_size

        for point in lidar_points:
            x, y, z, intensity = point
            px = int((x + 50) * width / 100)
            py = int((y + 50) * height / 100)
            if 0 <= px < width and 0 <= py < height:
                color_intensity = min(255, int(255 * intensity))
                color = (color_intensity, 255 - color_intensity, 0)
                self.window.set_at((px, py), color)

        pygame.display.flip()

    def close(self):
        """
        Closes the Pygame window.
        """
        pygame.quit()


def main():
    """
    Entry point for running the BEV visualizer independently.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Initialize BEVVisualizer and BEVRenderer
    visualizer = BEVVisualizer(host='127.0.0.1', port=65433)
    renderer = BEVRenderer()

    try:
        # Establish a connection to the server
        visualizer.connect_to_server()
        renderer.initialize_window()
        logging.info("Starting BEV visualization...")
        running = True

        while running:
            # Fetch LiDAR data and render all sensors
            lidar_data_buffer = visualizer.fetch_lidar_data()
            if lidar_data_buffer:
                for sensor_id, lidar_points in lidar_data_buffer.items():
                    lidar_array = np.frombuffer(lidar_points, dtype=np.float32).reshape(-1, 4)
                    renderer.draw_lidar_points(lidar_array)

            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False  # Exit the loop when the user closes the window

    except KeyboardInterrupt:
        logging.info("BEV visualization stopped manually.")
    except Exception as e:
        logging.error(f"Error in BEV visualizer: {e}")
    finally:
        visualizer.close_connection()  # Ensure the connection is closed
        renderer.close()  # Close the renderer window
        logging.info("BEV visualizer terminated.")


if __name__ == "__main__":
    main()