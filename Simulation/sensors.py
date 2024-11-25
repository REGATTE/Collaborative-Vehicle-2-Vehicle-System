import carla
import logging
import numpy as np
from queue import Queue
from threading import Thread, Lock

"""
worker function: runs as a daemon thread and processed tasks from the queue, continuously processing LiDAR Data
thereby handling lidar_data_buffer and lidar_data_lock
"""

class Sensors:
    def __init__(self):
        # Initialize the LIDAR data buffer, lock, and processing queue
        self.lidar_data_buffer = {}
        self.lidar_data_lock = Lock()
        self.task_queue = Queue()

        # Start the worker thread for asynchronous processing
        self.worker_thread = Thread(target=self.worker, daemon=True)
        self.worker_thread.start()
    
    def worker(self):
        """
        Worker thread to process LIDAR data asynchronously from the task queue.
        """
        while True:
            try:
                vehicle_id, data = self.task_queue.get()
                logging.info(f"Processing LIDAR data for vehicle {vehicle_id}...")
                points = len(data)  # Example processing: Count points in LIDAR data

                # Update the buffer with the processed data
                with self.lidar_data_lock:
                    self.lidar_data_buffer[vehicle_id] = points

                logging.info(f"Processed LIDAR data for vehicle {vehicle_id}: {points} points.")
                self.task_queue.task_done()
            except Exception as e:
                logging.error(f"Error in worker thread: {e}")
                self.task_queue.task_done()

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
                    'range': 50, 'rotation_frequency': 20, 'channels': 64,
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

    def lidar_callback(self, data, vehicle_id, ego_vehicle, proximity_mapping):
        """
        Process LIDAR data for vehicles in proximity.
        """
        try:
            # Verify vehicle ID and proximity
            if not isinstance(vehicle_id, int):
                logging.error(f"Invalid vehicle ID type: {vehicle_id}. Skipping.")
                return

            vehicle_actor = proximity_mapping.world.get_actor(vehicle_id)
            nearby_vehicles = proximity_mapping.find_vehicles_in_radius(ego_vehicle, [vehicle_actor])
            if not nearby_vehicles:
                return  # Vehicle not in proximity; skip processing

            # Add the task to the queue for asynchronous processing
            self.task_queue.put((vehicle_id, data))
            logging.debug(f"Added LIDAR data processing task for vehicle {vehicle_id} to queue.")

        except Exception as e:
            logging.error(f"Error in LIDAR callback for vehicle {vehicle_id}: {e}")

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

    def attach_sensor_suite(self, world, vehicle, vehicle_label, attached_sensors, ego_vehicle, proximity_mapping):
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

                        # Attach the callback for LIDAR
                        lidar.listen(
                            lambda data: self.lidar_callback(data, vehicle.id, ego_vehicle, proximity_mapping)
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
