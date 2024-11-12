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
        
        sensors = [
            # Front Camera
            {
                'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 300, 'height': 200, 'fov': 100, 'id': 'Front'
            },
            # Left Camera
            {
                'type': 'sensor.camera.rgb', 'x': 0.0, 'y': -0.7, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0, 'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'
            },
            # Right Camera
            {
                'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.7, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,  'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'
            },
            # Back Camera
            {
                'type': 'sensor.camera.rgb', 'x': -0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,  'width': 300, 'height': 200, 'fov': 100, 'id': 'Back'
            },
            # LiDAR
            {
                'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'LIDAR'
            },
            # GPS
            {
                'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'
            }
        ]
        
        return sensors
