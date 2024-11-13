from utils.config.config_loader import load_config

config_path = "utils/config/config.yaml"

try:
    config = load_config(config_path)
except Exception as e:
    print("Failed to load configuration. Exiting program.")
    exit(1)  # Exit program if config loading fails

class Sensors_Setup:
    def __init__(self):
        pass
    
    def sensors(self):
        """
        Define the sensor suite required by all the vehicles.
        
        All the smart vehicles will have the same sensor suite, to keep processing minimal.
        
        Sensor Suite:
            4x Camera: 1 camera on each side
            1x LiDAR: 1 main 360 lidar on the top
            1x GPS
        """
        
        return [
            # Front Camera
            {
                'type': 'sensor.camera.rgb',
                'x': 0.7,
                'y': 0.0,
                'z': config.sensors.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'width': config.sensors.camera_width,
                'height': config.sensors.camera_height,
                'fov': config.sensors.camera_fov,
                'id': 'Front_Camera',
            },
            # Left Camera
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0,
                'y': -0.7,
                'z': config.sensors.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': -90.0,
                'width': config.sensors.camera_width,
                'height': config.sensors.camera_height,
                'fov': config.sensors.camera_fov,
                'id': 'Left_Camera',
            },
            # Right Camera
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0,
                'y': 0.7,
                'z': config.sensors.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 90.0,
                'width': config.sensors.camera_width,
                'height': config.sensors.camera_height,
                'fov': config.sensors.camera_fov,
                'id': 'Right_Camera',
            },
            # Back Camera
            {
                'type': 'sensor.camera.rgb',
                'x': -0.7,
                'y': 0.0,
                'z': config.sensors.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 180.0,
                'width': config.sensors.camera_width,
                'height': config.sensors.camera_height,
                'fov': config.sensors.camera_fov,
                'id': 'Back_Camera',
            },
            # LiDAR
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 0.0,
                'y': 0.0,
                'z': config.sensors.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'id': 'LIDAR',
            },
            # GPS
            {
                'type': 'sensor.other.gnss',
                'x': 0.7,
                'y': -0.4,
                'z': config.sensors.z,
                'id': 'GPS',
            },
        ]
