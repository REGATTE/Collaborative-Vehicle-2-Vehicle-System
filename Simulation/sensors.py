import carla
import logging

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
        ]
        return sensors
    
    def attach_camera(self, world, vehicle, camera_config):
        """
        Attaches a camera to the vehicle.
        :param world: The CARLA world object.
        :param vehicle: The CARLA vehicle actor.
        :param camera_config: Configuration dictionary for the camera.
        :return: The camera sensor actor.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(camera_config['width']))
            camera_bp.set_attribute('image_size_y', str(camera_config['height']))
            camera_bp.set_attribute('fov', str(camera_config['fov']))
            
            sensor_transform = carla.Transform(
                carla.Location(x=camera_config['x'], y=camera_config['y'], z=camera_config['z']),
                carla.Rotation(roll=camera_config['roll'], pitch=camera_config['pitch'], yaw=camera_config['yaw'])
            )

            camera = world.spawn_actor(camera_bp, sensor_transform, attach_to=vehicle)
            logging.info(f"Attached Camera with ID: {camera.id} to Vehicle ID: {vehicle.id}")
            return camera
        except Exception as e:
            logging.error(f"Error attaching camera to Vehicle ID: {vehicle.id}: {e}")
            return None

    def attach_lidar(self, world, vehicle, label):
        """
        Attaches a LiDAR sensor to the vehicle. 
        :param world: The CARLA world object.
        :param vehicle: The CARLA vehicle actor.
        :param label: The CARLA vehicle label.
        :return: The LiDAR sensor actor.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('range', '50')
            lidar_bp.set_attribute('points_per_second', '56000')
            lidar_bp.set_attribute('rotation_frequency', '10')

            # Attach the LiDAR sensor to the vehicle
            lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            logging.info(f"LiDAR attached to {label} (Vehicle ID: {vehicle.id})")
            return lidar
        except Exception as e:
            logging.error(f"Error attaching LiDAR to {label} (Vehicle ID: {vehicle.id}): {e}")
            return None


    def attach_gnss(self, world, vehicle, gnss_config):
        """
        Attaches a GNSS sensor to the vehicle.
        :param world: The CARLA world object.
        :param vehicle: The CARLA vehicle actor.
        :param gnss_config: Configuration dictionary for the GNSS sensor.
        :return: The GNSS sensor actor.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            gnss_bp = blueprint_library.find('sensor.other.gnss')
            
            sensor_transform = carla.Transform(
                carla.Location(x=gnss_config['x'], y=gnss_config['y'], z=gnss_config['z'])
            )

            gnss = world.spawn_actor(gnss_bp, sensor_transform, attach_to=vehicle)
            logging.info(f"Attached GNSS with ID: {gnss.id} to Vehicle ID: {vehicle.id}")
            return gnss
        except Exception as e:
            logging.error(f"Error attaching GNSS to Vehicle ID: {vehicle.id}: {e}")
            return None

    def attach_sensor_suite(self, world, vehicle, vehicle_label):
        """
        Attaches a full sensor suite (Camera, LiDAR, GNSS) to the vehicle.
        :param world: The CARLA world object.
        :param vehicle: The CARLA vehicle actor.
        :param vehicle_label: Label for the vehicle (e.g., "ego_veh").
        :return: List of attached sensor actors.
        """
        sensors_config = self.sensor_suite()
        attached_sensors = []

        try:
            for sensor_config in sensors_config:
                sensor_type = sensor_config['type']
                logging.debug(f"Attempting to attach {sensor_type} to {vehicle_label} (ID: {vehicle.id})")

                if sensor_type == 'sensor.camera.rgb':
                    camera = self.attach_camera(world, vehicle, sensor_config)
                    if camera:
                        attached_sensors.append(camera)
                        logging.info(f"Attached Camera (ID: {camera.id}) to {vehicle_label} (ID: {vehicle.id})")

                elif sensor_type == 'sensor.lidar.ray_cast':
                    lidar = self.attach_lidar(world, vehicle, sensor_config)
                    if lidar:
                        attached_sensors.append(lidar)
                        logging.info(f"Attached LiDAR (ID: {lidar.id}) to {vehicle_label} (ID: {vehicle.id})")

                elif sensor_type == 'sensor.other.gnss':
                    gnss = self.attach_gnss(world, vehicle, sensor_config)
                    if gnss:
                        attached_sensors.append(gnss)
                        logging.info(f"Attached GNSS (ID: {gnss.id}) to {vehicle_label} (ID: {vehicle.id})")

            logging.info(f"[{vehicle_label}] Total sensors attached: {len(attached_sensors)}")
        except Exception as e:
            logging.error(f"Error attaching sensor suite to {vehicle_label}: {e}")

        return attached_sensors
