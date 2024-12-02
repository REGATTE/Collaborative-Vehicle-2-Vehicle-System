import carla
import logging
import numpy as np

class Sensors:
    def __init__(self):
        # Initialization logic if needed
        pass

    def sensor_suite(self):  # pylint: disable=no-self-use
        """
        Returns the configuration for the sensor suite.
        """
        sensors = [
            # Cameras
            {
                'type': 'sensor.camera.rgb',
                'transform': {'x': 0.7, 'y': -0.4, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'attributes': {'width': 1280, 'height': 720, 'fov': 100},
                'id': 'Left'
            },
            {
                'type': 'sensor.camera.rgb',
                'transform': {'x': 0.7, 'y': 0.4, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'attributes': {'width': 1280, 'height': 720, 'fov': 100},
                'id': 'Right'
            },
            {
                'type': 'sensor.camera.rgb',
                'transform': {'x': 0.7, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'attributes': {'width': 1280, 'height': 720, 'fov': 100},
                'id': 'Center'
            },
            # LiDAR
            {
                'type': 'sensor.lidar.ray_cast',
                'transform': {'x': 0.7, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'attributes': {
                    'range': 100, 'rotation_frequency': 20, 'channels': 128,
                    'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000
                },
                'id': 'LIDAR'
            },
            # GNSS
            {
                'type': 'sensor.other.gnss',
                'transform': {'x': 0.7, 'y': -0.4, 'z': 1.6},
                'id': 'GPS'
            }
        ]
        return sensors

    def attach_camera(self, world, vehicle, camera_config, transform):
        """
        Attaches the above camera's to the vehicle.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            camera_bp = blueprint_library.find(camera_config['type'])

            # Ensure attributes are set only if they exist in the blueprint
            attributes = camera_config.get('attributes', {})
            for key, value in attributes.items():
                if camera_bp.has_attribute(key):
                    camera_bp.set_attribute(key, str(value))
                else:
                    logging.warning(f"Attribute '{key}' not found in camera blueprint.")

            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            logging.info(f"Attached Camera with ID: {camera.id} to Vehicle ID: {vehicle.id}")
            return camera
        except Exception as e:
            logging.error(f"Error attaching camera to Vehicle ID: {vehicle.id}: {e}")
            return None
    
    def camera_callback(self, data, sensor_id, camera_data_buffer, camera_data_lock):
        """
        Callback function to process camera data, keyed by sensor ID.
        :param data: The raw camera data.
        :param sensor_id: The ID of the camera sensor.
        :param camera_data_buffer: Shared buffer to store processed camera frames.
        :param camera_data_lock: Thread lock for thread-safe access to the buffer.
        """
        try:
            # Convert raw data to a NumPy array
            frame = np.array(data.raw_data).reshape((data.height, data.width, 4))[:, :, :3]  # Extract RGB channels
            # logging.info(f"Processed camera frame for Sensor ID {sensor_id}.")

            # Store processed frame in the buffer
            with camera_data_lock:
                camera_data_buffer[sensor_id] = frame

        except Exception as e:
            logging.error(f"Error in camera_callback for Sensor ID {sensor_id}: {e}")


    def attach_lidar(self, world, vehicle, lidar_config, transform):
        """
        Attaches the above LiDAR sensor to the vehicle.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            lidar_bp = blueprint_library.find(lidar_config['type'])
            for key, value in lidar_config.get('attributes', {}).items():
                lidar_bp.set_attribute(key, str(value))
            lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
            logging.info(f"LiDAR attached with ID: {lidar.id} to Vehicle ID: {vehicle.id}")
            return lidar
        except Exception as e:
            logging.error(f"Error attaching LiDAR to Vehicle ID: {vehicle.id}: {e}")
            return None
        
    def ego_lidar_callback(self, data, sensor_id, lidar_data_buffer, lidar_data_lock):
        """
        Callback function to process LIDAR data for the ego vehicle, keyed by sensor ID.
        """
        try:
            # Convert memoryview to bytes to make it pickle-able
            lidar_data_bytes = bytes(data.raw_data)
            with lidar_data_lock:
                # Store raw LIDAR data, keyed by sensor ID
                lidar_data_buffer[sensor_id] = lidar_data_bytes # Use sensor ID as the key
                # logging.info(f"Ego LIDAR data stored for Sensor ID {sensor_id}: {len(points)} bytes.")
        except Exception as e:
            logging.error(f"Error in ego_lidar_callback for Sensor ID {sensor_id}: {e}")
        except AttributeError as e:
            logging.error(f"AttributeError in eg_lidar_callback for Sensor ID {sensor_id}: {e}")
        except KeyError as e:
            logging.error(f"KeyError in eg_lidar_callback for Sensor ID {sensor_id}: {e}")
        except Exception as e:
            logging.error(f"Unhandled error in eg_lidar_callback for Sensor ID {sensor_id}: {e}")

    def lidar_callback(self, data, vehicle_id, sensor_id, ego_vehicle, proximity_mapping, lidar_data_buffer, lidar_data_lock):
        """
        Process LIDAR data for sensors in proximity, keyed by sensor ID.
        """
        try:
            with lidar_data_lock:
                # Ensure sensor ID is valid
                if not isinstance(sensor_id, int):
                    logging.error(f"Invalid sensor ID: {sensor_id}. Skipping.")
                    return

                # Check if the vehicle associated with this sensor is in proximity
                vehicle_actor = proximity_mapping.world.get_actor(vehicle_id)
                if not vehicle_actor:
                    logging.error(f"No actor found for vehicle ID: {vehicle_id}. Skipping.")
                    return
                # Check if the vehicle associated with this sensor is in proximity
                vehicles_in_radius = proximity_mapping.find_vehicles_in_radius(ego_vehicle, [vehicle_actor])
                if vehicle_id not in vehicles_in_radius:
                    #logging.debug(f"Vehicle ID {vehicle_id} is not in proximity. Ignoring LIDAR data.")
                    return
                # Process and store LIDAR data
                lidar_data_bytes = bytes(data.raw_data)  # Access raw LIDAR data
                lidar_data_buffer[sensor_id] = lidar_data_bytes  # Key buffer by sensor ID
                # Log the processed data
                logging.info(
                    # f"LIDAR data buffer: {lidar_data_buffer}"
                    # f"Processed LIDAR data for Sensor ID {sensor_id} "
                    f"(Vehicle ID {vehicle_id}): {len(lidar_data_bytes)} points. "
                    f"Distance to Ego: {vehicles_in_radius[vehicle_id][1]:.2f}m."
                )

        except AttributeError as e:
            logging.error(f"AttributeError in LIDAR callback for Sensor ID {sensor_id}: {e}")
        except KeyError as e:
            logging.error(f"KeyError in LIDAR callback for Sensor ID {sensor_id}: {e}")
        except Exception as e:
            logging.error(f"Unhandled error in LIDAR callback for Sensor ID {sensor_id}: {e}")

    def attach_gnss(self, world, vehicle, gnss_config, transform):
        """
        Attaches a GNSS sensor to the vehicle.
        """
        try:
            blueprint_library = world.get_blueprint_library()
            gnss_bp = blueprint_library.find(gnss_config['type'])
            gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
            logging.info(f"Attached GNSS with ID: {gnss.id} to Vehicle ID: {vehicle.id}")
            return gnss
        except Exception as e:
            logging.error(f"Error attaching GNSS to Vehicle ID: {vehicle.id}: {e}")
            return None

    def attach_sensor_suite(self, world, vehicle, vehicle_label, lidar_data_buffer, lidar_data_lock, camera_data_buffer, camera_data_lock, attached_sensors, ego_vehicle, proximity_mapping):
        """
        Attaches a full sensor suite (Camera, LiDAR, GNSS) to the vehicle.
        """
        sensors_config = self.sensor_suite()
        vehicle_sensors = []
        blueprint_library = world.get_blueprint_library()

        for sensor_config in sensors_config:
            try:
                # Validate sensor configuration
                if 'type' not in sensor_config or 'transform' not in sensor_config:
                    logging.warning(f"Invalid sensor configuration: {sensor_config}")
                    continue

                sensor_type = sensor_config['type']
                transform_config = sensor_config['transform']
                transform = carla.Transform(
                    carla.Location(x=transform_config['x'], y=transform_config['y'], z=transform_config['z']),
                    carla.Rotation(
                        roll=transform_config.get('roll', 0.0),
                        pitch=transform_config.get('pitch', 0.0),
                        yaw=transform_config.get('yaw', 0.0),
                    )
                )

                if sensor_type == "sensor.lidar.ray_cast":
                    # Attach the LIDAR sensor
                    lidar_bp = blueprint_library.find(sensor_config['type'])
                    lidar_bp.set_attribute('range', '50')  # Set LIDAR range
                    lidar_bp.set_attribute('rotation_frequency', '20')
                    lidar_bp.set_attribute('channels', '32')
                    lidar_bp.set_attribute('points_per_second', '56000')

                    # Spawn the LIDAR sensor and attach to the vehicle
                    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
                    if lidar:
                        vehicle_sensors.append(lidar)
                        attached_sensors.append(lidar)
                        logging.info(f"LIDAR attached with ID: {lidar.id} to Vehicle ID: {vehicle.id}")

                        # if ego_Vehicle
                        if vehicle_label == 'ego_veh':
                            lidar.listen(
                                lambda data: self.ego_lidar_callback(data, 32, lidar_data_buffer, lidar_data_lock)
                            )
                        else:
                            # Attach the callback for LIDAR
                            lidar.listen(
                                lambda data: self.lidar_callback(data, vehicle.id, lidar.id , ego_vehicle, proximity_mapping, lidar_data_buffer, lidar_data_lock)
                            )
                    else:
                        logging.warning(f"Failed to attach LIDAR to Vehicle ID: {vehicle.id}")

                elif sensor_type == "sensor.camera.rgb":
                    # Retrieve the camera blueprint
                    camera_bp = blueprint_library.find(sensor_config['type'])

                    # Ensure image size attributes are set
                    if camera_bp.has_attribute('image_size_x') and camera_bp.has_attribute('image_size_y'):
                        camera_bp.set_attribute('image_size_x', '1920')  # Example resolution
                        camera_bp.set_attribute('image_size_y', '1080')
                    else:
                        logging.warning("Camera blueprint does not support resolution attributes.")

                    # Set other attributes (if required, e.g., FOV)
                    if camera_bp.has_attribute('fov'):
                        camera_bp.set_attribute('fov', '90')

                    # Spawn the camera and attach it to the vehicle
                    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                    if camera:
                        vehicle_sensors.append(camera)
                        attached_sensors.append(camera)
                        logging.info(f"Camera attached with ID: {camera.id} to Vehicle ID: {vehicle.id}")

                        # Attach the callback for the 3rd sensor (Center Camera)
                        if sensor_config.get('id') == 'Center':
                            camera.listen(
                                lambda data: self.camera_callback(data, 31, camera_data_buffer, camera_data_lock)  # Assuming 31 is the sensor ID
                            )
                    else:
                        logging.warning(f"Failed to attach camera to Vehicle ID: {vehicle.id}")


                elif sensor_type == "sensor.other.gnss":
                    # Attach GNSS sensor
                    gnss = self.attach_gnss(world, vehicle, sensor_config, transform)
                    if gnss:
                        vehicle_sensors.append(gnss)
                        attached_sensors.append(gnss)

            except Exception as e:
                logging.error(f"Error attaching {sensor_type} to {vehicle_label}: {e}")

        logging.info(f"[{vehicle_label}] Total sensors successfully attached: {len(vehicle_sensors)}")
        return vehicle_sensors
