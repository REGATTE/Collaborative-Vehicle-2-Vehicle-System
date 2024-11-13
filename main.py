import carla
import pygame
import time
from Simulation.ego_vehicle import EgoVehicle
from Simulation.generate_traffic import TrafficManager
from Simulation.sensors import Sensors_Setup
from agents.navigation.basic_agent import BasicAgent

from utils.config.config_loader import load_config

config_path = "utils/config/config.yaml"

try:
    config = load_config(config_path)
except Exception as e:
    print("Failed to load configuration. Exiting program.")
    exit(1)  # Exit program if config loading fails

class CarlaManager:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.sensors = []
        self.display = None
        self.camera_sensor = None
        self.ego_vehicle = None
        self.traffic_manager = TrafficManager(self.world, self.blueprint_library)

    def setup_display(self):
        pygame.init()
        self.display = pygame.display.set_mode((config.pygame_display.width, config.pygame_display.height))
        pygame.display.set_caption("CARLA Simulation")
        print("Pygame display initialized.")

    def spawn_ego_vehicle(self):
        spawn_point = self.world.get_map().get_spawn_points()[0]
        ego_vehicle_bp = self.blueprint_library.filter(config.simulated_vehicles.ego_vehicle)[0]
        self.ego_vehicle = EgoVehicle(self.world, ego_vehicle_bp, spawn_point)
        print("Ego vehicle spawned.")

    def run_simulation(self):
        self.setup_display()
        self.spawn_ego_vehicle()
        self.traffic_manager.spawn_smart_vehicles(config.simulated_vehicles.smp)  # Spawn smart vehicles
        self.traffic_manager.spawn_random_traffic(config.simulated_vehicles.rvt)  # Spawn random vehicles

        clock = pygame.time.Clock()
        try:
            while True:
                clock.tick(30)  # Limit the frame rate to 30 FPS
                # Update the ego vehicle, which will receive data from nearby smart vehicles
                self.ego_vehicle.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return  # Exit the loop if the window is closed
        finally:
            self.cleanup()
            pygame.quit()

    def cleanup(self):
        # Clean up all vehicles and actors
        self.ego_vehicle.destroy()
        self.traffic_manager.destroy_all()
        print("Cleaned up all actors.")

if __name__ == '__main__':
    carla_manager = CarlaManager()
    carla_manager.run_simulation()
