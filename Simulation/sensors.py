class Sensors:
    def __init__(self):
        # Add your initialization logic here
        pass

    def sensor_suite(self):  # pylint: disable=no-self-use
        
        sensors = [
            #Camera
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 1280, 'height': 720, 'fov': 100, 'id': 'Center'},
            #Lidar
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'range': 50, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000, 'id': 'LIDAR'},
            #GPS
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            # Depth Camera
            {'type': 'sensor.camera.depth', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 1280, 'height': 720, 'fov': 100, 'id': 'DepthCamera'}
        ]
        return sensors