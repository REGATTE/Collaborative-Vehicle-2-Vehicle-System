import carla
import random
from sensors import Sensors_Setup

class TrafficManager:
    def __init__(self, world, blueprint_library):
        self.world = world
        self.blueprint_library = blueprint_library
        self.vehicles = []

    def spawn_smart_vehicles(self, count):
        spawn_points = self.world.get_map().get_spawn_points()
        for i in range(count):
            vehicle_bp = self.blueprint_library.filter('vehicle.audi.tt')[0]  # Example vehicle type
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.vehicles.append(vehicle)
            # Add sensor suite to smart vehicles
            sensor_setup = Sensors_Setup().sensors()
            for sensor_spec in sensor_setup:
                sensor_bp = self.blueprint_library.find(sensor_spec['type'])
                sensor_transform = carla.Transform(
                    carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z']),
                    carla.Rotation(roll=sensor_spec['roll'], pitch=sensor_spec['pitch'], yaw=sensor_spec['yaw'])
                )
                sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
                sensor.listen(lambda data: self.share_data_with_ego(data, vehicle))

    def spawn_random_traffic(self, count):
        spawn_points = self.world.get_map().get_spawn_points()
        for i in range(count):
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.vehicles.append(vehicle)

    def share_data_with_ego(self, data, vehicle):
        # Logic to share sensor data with the ego vehicle
        print(f"Sharing data from vehicle {vehicle.id} with ego vehicle.")

    def destroy_all(self):
        for vehicle in self.vehicles:
            vehicle.destroy()
        self.vehicles.clear()
