import carla
from agents.navigation.basic_agent import BasicAgent
from utils import data_processing_utils
from sensors import Sensors_Setup

class EgoVehicle:
    def __init__(self, world, vehicle_bp, spawn_point):
        self.vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        self.agent = BasicAgent(self.vehicle)
        self.world = world
        self.sensors = []
        self.setup_sensors()

    def setup_sensors(self):
        sensor_setup = Sensors_Setup().sensors()
        for sensor_spec in sensor_setup:
            sensor_bp = self.world.get_blueprint_library().find(sensor_spec['type'])
            sensor_transform = carla.Transform(
                carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z']),
                carla.Rotation(roll=sensor_spec['roll'], pitch=sensor_spec['pitch'], yaw=sensor_spec['yaw'])
            )
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.vehicle)
            self.sensors.append(sensor)
            sensor.listen(lambda data: self.process_sensor_data(data, sensor_spec['id']))

    def process_sensor_data(self, data, sensor_id):
        """
        Only focussing on LiDAR and GPS.
        """
        processed_data = data_processing_utils.process_data(data)  # Utilize data processing utility
        if sensor_id == 'LIDAR':
            self.agent.handle_lidar_data(processed_data)  # Example: lidar data processing
        elif sensor_id == 'GPS':
            self.agent.handle_gps_data(processed_data)  # Example: GPS data processing
        # Additional sensor data processing as needed

    def receive_shared_data(self, data):
        # Logic to incorporate shared data from nearby vehicles
        print("Received shared data from nearby vehicles")

    def update(self):
        # Check for nearby vehicles and request shared data if close
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id != self.vehicle.id and self.vehicle.get_location().distance(vehicle.get_location()) < 20:
                self.receive_shared_data(vehicle)

    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()
        self.vehicle.destroy()
