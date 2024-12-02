import logging
import socket
import pickle
import numpy as np
import pygame


class BEVVisualizer:
    """
    Handles the connection to the LiDAR data server and fetches data for visualization.
    """

    def __init__(self, host='127.0.0.1', ego_port=65433, combined_port=65434):
        """
        Initialize the BEVVisualizer with server connection details.
        :param host: Host address of the LiDAR data servers.
        :param ego_port: Port number for ego LiDAR data.
        :param combined_port: Port number for combined LiDAR data.
        """
        self.host = host
        self.ego_port = ego_port
        self.combined_port = combined_port
        self.ego_socket = None

    def connect_to_ego_server(self):
        """
        Establish a persistent connection to the ego LiDAR data server.
        """
        self.ego_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ego_socket.connect((self.host, self.ego_port))
        logging.info(f"Connected to Ego LiDAR Data Server at {self.host}:{self.ego_port}")

    def fetch_ego_lidar_data(self):
        """
        Fetch the latest ego LiDAR data buffer from the server.
        :return: Dictionary containing LiDAR data from all sensors or an empty dictionary if no data is available.
        """
        try:
            data = b""
            while True:
                chunk = self.ego_socket.recv(4096)
                if b"<END>" in chunk:
                    data += chunk.split(b"<END>")[0]
                    break
                data += chunk

            lidar_data_buffer = pickle.loads(data)  # Deserialize the data

            if not lidar_data_buffer:
                logging.warning("No ego LiDAR data received from server.")
                return {}

            logging.info(f"Fetched ego LiDAR data for {len(lidar_data_buffer.keys())} sensors.")
            return lidar_data_buffer

        except Exception as e:
            logging.error(f"Error fetching ego LiDAR data: {e}")
            return {}

    def fetch_combined_lidar_data(self):
        """
        Fetch the combined LIDAR data sent as a one-off TCP connection.
        :return: NumPy array of combined LIDAR points or None if no data is available.
        """
        try:
            # Create a one-off TCP connection for combined data
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.settimeout(0.1)  # Set a timeout to avoid blocking
                client_socket.bind((self.host, self.combined_port))
                client_socket.listen(1)  # Listen for one connection at a time
                try:
                    conn, addr = client_socket.accept()
                    logging.info(f"Connection established with {addr}.")

                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if b"<END>" in chunk:
                            data += chunk.split(b"<END>")[0]
                            break
                        data += chunk

                    combined_lidar_data = pickle.loads(data)  # Deserialize the data
                    logging.info(f"Received combined LIDAR data with {len(combined_lidar_data)} points.")
                    return np.array(combined_lidar_data, dtype=np.float32)
                except socket.timeout:
                    return None  # No data received within timeout

        except Exception as e:
            logging.error(f"Error receiving combined LIDAR data: {e}")
            return None

    def close_connection(self):
        """
        Close the persistent connection to the ego LiDAR data server.
        """
        if self.ego_socket:
            self.ego_socket.close()
            logging.info("Disconnected from Ego LiDAR Data Server.")

class BEVRenderer:
    """
    Renders LiDAR data in a Bird's Eye View (BEV) using PyGame.
    """

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.screen = None

    def initialize_window(self):
        """
        Initializes the PyGame window for rendering.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Bird's Eye View Visualization")
        logging.info("PyGame window initialized.")

    def draw_lidar_points(self, lidar_array):
        """
        Draws LiDAR points on the PyGame window, flipped vertically.
        :param lidar_array: NumPy array of LiDAR points (x, y, z, intensity).
        """
        self.screen.fill((0, 0, 0))  # Clear the screen with black
        for point in lidar_array:
            x, y = int(point[0] * 10 + self.width // 2), int(self.height // 2 + point[1] * 10)
            pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 2)  # Green points
        pygame.display.flip()

    def draw_combined_lidar_points(self, combined_lidar_array):
        """
        Draws combined LiDAR points on the PyGame window, flipped vertically.
        :param combined_lidar_array: NumPy array of combined LiDAR points.
        """
        self.screen.fill((0, 0, 0))  # Clear the screen with black
        for point in combined_lidar_array:
            x, y = int(point[0] * 10 + self.width // 2), int(self.height // 2 + point[1] * 10)
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 2)  # Red points for combined data
        pygame.display.flip()

    def close(self):
        """
        Closes the PyGame window.
        """
        pygame.quit()
        logging.info("PyGame window closed.")


def main():
    """
    Entry point for running the BEV visualizer independently.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Initialize BEVVisualizer and BEVRenderer
    visualizer = BEVVisualizer(host='127.0.0.1', ego_port=65433, combined_port=65434)
    renderer = BEVRenderer()

    try:
        # Establish connection for ego LiDAR data
        visualizer.connect_to_ego_server()
        renderer.initialize_window()
        logging.info("Starting BEV visualization...")
        running = True

        while running:
            # Fetch ego LiDAR data and render it
            lidar_data_buffer = visualizer.fetch_ego_lidar_data()
            if lidar_data_buffer:
                for sensor_id, lidar_points in lidar_data_buffer.items():
                    lidar_array = np.frombuffer(lidar_points, dtype=np.float32).reshape(-1, 4)
                    renderer.draw_lidar_points(lidar_array)

            # Check for combined LiDAR data
            combined_lidar_data = visualizer.fetch_combined_lidar_data()
            if combined_lidar_data is not None and combined_lidar_data.size > 0:
                logging.info("Rendering combined LIDAR data.")
                renderer.draw_combined_lidar_points(combined_lidar_data)

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
