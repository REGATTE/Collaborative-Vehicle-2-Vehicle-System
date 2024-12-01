import carla
import random
import pygame
import numpy as np
import logging
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import os
import json

from agents.controller import ControlObject
from agents.EnvironmentManager import EnvironmentManager
from agents.waypoint_manager import WaypointManager
from Simulation.generate_traffic import setup_traffic_manager, spawn_vehicles, spawn_walkers, cleanup
from Simulation.sensors import Sensors
from Simulation.ego_vehicle import EgoVehicleListener
from Simulation.PathPlanning.obstacle_detection import DataFusion
from Simulation.PathPlanning.occupancy_grid_mapping import OccupancyGridMap, OverlayGridOverlay
from utils.config.config_loader import load_config
from utils.logging_config import configure_logging
from utils.carla_utils import initialize_carla, setup_synchronous_mode
from utils.vehicle_mapping.vehicle_mapping import save_vehicle_mapping, load_vehicle_mapping
from utils.proximity_mapping import ProximityMapping
from utils.bbox.bbox import BoundingBoxExtractor
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from PIL import Image
import cv2

class RenderObject:
    """
    Handles rendering images from the camera to a PyGame surface.
    """
    def __init__(self, width, height):
        """
        Initializes the RenderObject with a surface and dimensions.
        :param width: The width of the PyGame surface.
        :param height: The height of the PyGame surface.
        """
        self.width = width
        self.height = height
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


def initialize_pygame(camera_bp, num_labels, fixed_resolution=(1920, 1080)):
    """
    Initializes the PyGame window with a fixed resolution or dynamic calculation.
    :param camera_bp: Blueprint of the camera.
    :param num_labels: Number of labels to display in the menu bar.
    :param fixed_resolution: Fixed resolution for the PyGame window.
    :return: PyGame display surface and adjusted resolution (width, height).
    """
    pygame.init()

    # Use the fixed resolution directly
    window_width, window_height = fixed_resolution
    game_display = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    game_display.fill((0, 0, 0))
    pygame.display.flip()

    return game_display, window_width, window_height

def pygame_callback(image, render_object):
    """
    Callback function to process camera images and render them in PyGame.
    :param image: Camera image from CARLA.
    :param render_object: RenderObject to display the image.
    """
    try:
        # Convert raw image data to RGB format
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha channel
        array = array[:, :, ::-1]  # Convert BGRA to RGB

        # Scale the image to fit the render object's surface dimensions
        scaled_image = pygame.transform.scale(
            pygame.surfarray.make_surface(array.swapaxes(0, 1)),
            (render_object.width, render_object.height)
        )

        # Update the render surface
        render_object.surface = scaled_image
    except Exception as e:
        logging.error(f"Error in pygame_callback: {e}")

def attach_follow_camera(world, vehicle, camera_transform):
    """
    Attaches a camera sensor to a vehicle with a specified transform.
    :param world: CARLA world object.
    :param vehicle: CARLA vehicle actor to attach the camera to.
    :param camera_transform: Transform object specifying the camera's position and orientation.
    :return: The attached camera actor and its blueprint.
    """
    try:
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')

        # Spawn the camera sensor
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logging.info(f"Car following camera attached to Vehicle ID: {vehicle.id}")
        return camera, camera_bp
    except Exception as e:
        logging.error(f"Error attaching camera to Vehicle ID: {vehicle.id}: {e}")
        return None, None

def get_ego_vehicle_camera_frame(vehicle_mapping_path, camera_data_buffer, camera_data_lock):
    """
    Retrieve the latest camera frame from the ego vehicle's camera sensor.
    :param vehicle_mapping_path: Path to the vehicle_mapping.json file.
    :param camera_data_buffer: Shared buffer containing camera data from all sensors.
    :param camera_data_lock: Thread lock for accessing the buffer safely.
    :return: Numpy array containing the camera frame or None if no data is available.
    """
    try:
        # Load the vehicle mapping from the JSON file
        with open(vehicle_mapping_path, "r") as f:
            vehicle_mapping = json.load(f)
        
        # Retrieve the camera sensor ID (3rd sensor)
        camera_sensor_id = vehicle_mapping.get("ego_veh", {}).get("sensors", [])[2]
        if not camera_sensor_id:
            logging.warning("Camera sensor ID not found in vehicle mapping.")
            return None

        logging.info(f"Attempting to access camera data for Sensor ID {camera_sensor_id}.")

        # Access the camera data buffer
        with camera_data_lock:
            processed_camera_frame = camera_data_buffer.get(camera_sensor_id, None)

        if processed_camera_frame is None:
            logging.warning(f"No camera data available for Sensor ID {camera_sensor_id}.")
            return None

        logging.info("Camera frame retrieved successfully.")
        return processed_camera_frame

    except FileNotFoundError:
        logging.error(f"Vehicle mapping file not found: {vehicle_mapping_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from vehicle mapping: {e}")
        return None
    except Exception as e:
        logging.error(f"Error retrieving camera frame: {e}")
        return None

def get_ego_vehicle_lidar_data(lidar_data_buffer, lidar_data_lock):
    """
    Retrieve and process the LIDAR data for the ego vehicle from the lidar_data_buffer.
    :param lidar_data_buffer: Shared buffer containing LIDAR data from all sensors.
    :param lidar_data_lock: Thread lock for accessing the buffer safely.
    :return: Numpy array containing processed LIDAR points ([x, y, z, i]) or an empty array if data is unavailable.
    """
    try:
        lidar_sensor_id = 32  # Sensor key for the ego vehicle's LIDAR
        logging.info(f"Attempting to access LIDAR data for Sensor ID {lidar_sensor_id}.")

        # Access the LiDAR data from the buffer
        with lidar_data_lock:
            raw_lidar_data = lidar_data_buffer.get(lidar_sensor_id, [])

        if not raw_lidar_data:
            logging.warning(f"No LIDAR data available for Sensor ID {lidar_sensor_id}.")
            return np.array([])  # Return an empty array if no data is found

        # Convert raw data to a NumPy array
        if isinstance(raw_lidar_data, memoryview):
            logging.info("Converting memoryview LIDAR data to list.")
            raw_lidar_data = list(raw_lidar_data)

        lidar_array = np.frombuffer(bytearray(raw_lidar_data), dtype=np.float32).reshape(-1, 4)
        logging.info(f"LIDAR data processed with {lidar_array.shape[0]} points.")
        return lidar_array

    except Exception as e:
        logging.error(f"Error retrieving LIDAR data: {e}")
        return np.array([])  # Return an empty array in case of error
    
def preprocess_fused_results(fused_results):
    """
    Preprocess the fused results to extract obstacle positions.
    :param fused_results: List of detection results from DataFusion.
    :return: List of obstacle positions [(x, y), ...].
    """
    obstacle_positions = []

    for result in fused_results:
        if "bbox" in result:
            # Extract bounding box center as obstacle position
            x_center = (result["bbox"][0] + result["bbox"][2]) / 2
            y_center = (result["bbox"][1] + result["bbox"][3]) / 2
            obstacle_positions.append((x_center, y_center))

    logging.debug(f"Preprocessed obstacle positions: {obstacle_positions}")
    return obstacle_positions

def obstacle_detection_worker(camera_frame, lidar_data, ego_vehicle, occupancy_grid, overlay_grid, frame_number, data_fusion):
    try:
        logging.info(f"Performing obstacle detection using DataFusion with {lidar_data.shape[0]} LiDAR points.")

        # Ensure DataFusion has access to updated combined LiDAR data
        if data_fusion.lidar_detector.combined_lidar_data is None:
            data_fusion.lidar_detector.update_combined_lidar_data(lidar_data)

        # Perform data fusion
        fused_results = data_fusion.fuse_data(camera_frame, lidar_data)
        logging.debug(f"Fused obstacle detection results: {fused_results}")

        # Fetch ego vehicle's location
        ego_vehicle_location = {
            "x": ego_vehicle.get_transform().location.x,
            "y": ego_vehicle.get_transform().location.y,
            "z": ego_vehicle.get_transform().location.z
        }

        # Preprocess fused results into obstacle positions
        obstacle_positions = preprocess_fused_results(fused_results)

        # Update occupancy grid map with obstacle positions
        occupancy_grid.update_with_obstacles(obstacle_positions, ego_vehicle_location)

        # Overlay and save the occupancy grid on the camera frame
        combined_image = overlay_grid.overlay_on_image(camera_frame)
        overlay_grid.save_frame(combined_image, frame_number)
        logging.info(f"Frame {frame_number} saved successfully with occupancy grid overlay.")
    except Exception as e:
        logging.error(f"Error in obstacle detection: {e}")

def game_loop(world, game_display, camera, render_object, control_object, vehicle_mapping, env_manager, 
              ego_vehicle, smart_vehicles, lidar_data_buffer, lidar_data_lock, camera_data_buffer, camera_data_lock,
              waypoint_manager,birdview_producer = None, data_fusion=None):
    """
    Main game loop for updating the CARLA world and PyGame display.
    """
    crashed = False
    font = pygame.font.SysFont('Arial', 16)
    menu_bar_height = 30
    width, height = game_display.get_size()

    proximity_mapping = ProximityMapping(world, radius=20.0)
    proximity_state = {}  # Proximity tracking state

    #Initialise thread pool and queue for obstacle detection
    executor = ThreadPoolExecutor(max_workers=2)
    obstacle_detection_queue = Queue(maxsize=5)

    # List of vehicle labels
    vehicle_keys = list(vehicle_mapping.keys())
    current_vehicle_index = 0
    active_vehicle_label = vehicle_keys[current_vehicle_index]

    bev_actor = ego_vehicle

    # Initialize Occupancy Grid Map and Overlay
    frame_number = 0
    grid_size = (500, 500)  # 500x500 grid
    cell_size = 0.1  # Each cell represents 0.1 meters
    world_size = (50, 50)  # World size is 50x50 meters
    occupancy_grid = OccupancyGridMap(grid_size, cell_size, world_size)
    overlay_grid = OverlayGridOverlay(occupancy_grid, cell_size, world_size, save_folder="frames/occ_map")

    def obstacle_worker():
        """Worker to process obstacle detection from the queue."""
        while not obstacle_detection_queue.empty():
            (camera_frame, lidar_data, ego_vehicle, occupancy_grid, overlay_grid, frame_number, data_fusion) = obstacle_detection_queue.get()
            try:
                obstacle_detection_worker(camera_frame, lidar_data, ego_vehicle, occupancy_grid, overlay_grid, frame_number, data_fusion)
            except Exception as e:
                logging.error(f"Error in obstacle detection worker: {e}")

    try:
        while not crashed:
            # Tick the simulation
            world.tick()

            # Clear the screen
            game_display.fill((0, 0, 0))

            # Render the camera feed
            if render_object.surface:
                game_display.blit(render_object.surface, (0, menu_bar_height))  # Offset for menu bar height

            # Render the menu bar
            env_manager.draw_vehicle_labels_menu_bar(game_display, font, vehicle_mapping, width, active_vehicle_label)

            # Proximity mapping and LIDAR logic (every 2nd frame)
            if frame_number%2 == 0:
                try:
                    proximity_mapping.log_proximity_and_trigger_communication(
                        ego_vehicle,
                        [smart_vehicle.id for smart_vehicle in smart_vehicles],
                        world,
                        proximity_state,
                        vehicle_mapping,
                        lidar_data_buffer,
                        lidar_data_lock
                    )
                except Exception as e:
                    logging.error(f"Error in proximity mapping: {e}")

            # Obstacle detection with data fusion
            # Fetch ego vehicle data
            try:
                camera_frame = get_ego_vehicle_camera_frame("utils/vehicle_mapping/vehicle_mapping.json", camera_data_buffer, camera_data_lock)
                lidar_data = get_ego_vehicle_lidar_data(lidar_data_buffer, lidar_data_lock)

                if camera_frame is not None and lidar_data is not None:
                    logging.info(f"Performing obstacle detection using DataFusion with {lidar_data.shape[0]} LiDAR points.")
                    # Add data to the queue
                    # Log when queue is full
                    if obstacle_detection_queue.full():
                        logging.warning("Obstacle detection queue has reached its maximum size. Skipping frame addition.")
                        obstacle_detection_queue.get_nowait()  # Remove the oldest frame to make space
                    
                    # Add data to the queue
                    obstacle_detection_queue.put((camera_frame, lidar_data, ego_vehicle, occupancy_grid, overlay_grid, frame_number, data_fusion))
                    logging.info(f"Added frame {frame_number} to obstacle detection queue. Current queue size: {obstacle_detection_queue.qsize()}")
                    
                    #start worker thread for obstacle detection
                    executor.submit(obstacle_detection_worker)

            except Exception as e:
                 logging.error(f"Error in obstacle detection setup: {e}")
            
            # Manage waypoints for smart vehicles
            try:
                waypoint_manager.manage_all_vehicles(periodic_update_interval=20)  # Update every 20 ticks
            except Exception as e:
                logging.error(f"Error in waypoint management: {e}")

            pygame.display.flip()  # Update the PyGame display
            control_object.process_control()

            #Birds eye view
            if birdview_producer :
                birdview = birdview_producer.produce(
                agent_vehicle=bev_actor  # carla.Actor (spawned vehicle)
                )
                rgb = BirdViewProducer.as_rgb(birdview)
                #cv2.imshow("BirdView RGB", rgb)
                pil_image = Image.fromarray(rgb)
                bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                cv2.imshow("BirdView BGR", bgr)
                cv2.waitKey(1)

            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        # Switch vehicle
                        try:
                            current_vehicle_index = (current_vehicle_index + 1) % len(vehicle_keys)
                            active_vehicle_label = vehicle_keys[current_vehicle_index]
                            logging.info(f"Switched to vehicle: {active_vehicle_label}")

                            # Reattach camera
                            new_vehicle = world.get_actor(vehicle_mapping[active_vehicle_label]["actor_id"])
                            if camera and camera.is_listening:
                                camera.stop()
                                camera.destroy()

                            camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
                            camera, camera_bp = attach_follow_camera(world, new_vehicle, camera_transform)
                            bev_actor = new_vehicle
                            if not camera:
                                logging.error(f"Failed to attach camera to {active_vehicle_label}.")
                                continue
                            camera.listen(lambda image: pygame_callback(image, render_object))
                        except Exception as e:
                            logging.error(f"Error switching to vehicle {active_vehicle_label}: {e}")

    except Exception as e:
        logging.error(f"An error occurred during the game loop: {e}")
    finally:
        # Cleanup camera
        if camera and camera.is_listening:
            camera.stop()
            camera.destroy()

        # Quit PyGame
        pygame.quit()
        logging.info("Game loop terminated.")

def main():
    """
    Entry point for the CARLA simulation.
    """
    configure_logging()
    logging.info("Starting CARLA simulation...")

    # Load configuration
    config = load_config("utils/config/config.yaml")
    logging.info("Configuration loaded successfully from utils/config/config.yaml")

    # Initialize CARLA client and world
    client, world = initialize_carla()
    logging.info("Client and world initialized successfully.")
    
    traffic_manager = client.get_trafficmanager()
    setup_synchronous_mode(world, traffic_manager, config)

    # Environment manager
    env_manager = EnvironmentManager(world)
    data_fusion_obs_det = DataFusion(vehicle_mapping_path="utils/vehicle_mapping/vehicle_mapping.json")

    # Initialize data buffers and tracking
    lidar_data_lock = Lock()
    camera_data_lock = Lock()
    attached_sensors = []
    lidar_data_buffer = {}
    camera_data_buffer = {}

    # Proximity mapping for LIDAR
    proximity_mapping = ProximityMapping(world, radius=20.0)

    # Initial cleanup
    env_manager.cleanup_existing_actors()

    # Spawn vehicles
    vehicles, spawn_locations = env_manager.spawn_with_retries(
        client, traffic_manager, config.simulation.num_vehicles, config.simulation.spawn_retries
    )
    ego_vehicle, smart_vehicles, vehicle_mapping = env_manager.designate_ego_and_smart_vehicles(
        vehicles, spawn_locations, world, config
    )

    # Cleanup after designation
    env_manager.cleanup_existing_actors(vehicle_mapping=vehicle_mapping)

    # Start the ego listener in a separate thread
    ego_listener = EgoVehicleListener(
        host='127.0.0.1', 
        port=65432, 
        ego_vehicle=ego_vehicle, 
        world=world,
        lidar_data_buffer=lidar_data_buffer,
        lidar_data_lock=lidar_data_lock
    )
    ego_listener_thread = threading.Thread(target=ego_listener.start_listener, daemon=True)
    ego_listener_thread.start()

    time.sleep(1)
    
    birdview_producer = None
    enableBev = config.simulation.bev_enable
    if enableBev :
        birdview_producer = BirdViewProducer(
        client,  # carla.Client
        target_size=PixelDimensions(width=150, height=336),
        pixels_per_meter=4,
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
        )

    # Attach sensors
    sensors = Sensors()
    ego_vehicle_sensors = sensors.attach_sensor_suite(
        world, 
        ego_vehicle, 
        "ego_veh", 
        lidar_data_buffer, 
        lidar_data_lock, 
        camera_data_buffer, 
        camera_data_lock, 
        attached_sensors, 
        ego_vehicle, 
        proximity_mapping)
    logging.info(f"Ego vehicle has {len(ego_vehicle_sensors)} sensors attached.")

    for idx, smart_vehicle in enumerate(smart_vehicles, start=1):
        vehicle_label = f"smart_veh_{idx}"
        smart_sensors = sensors.attach_sensor_suite(
            world, 
            smart_vehicle, 
            vehicle_label, 
            lidar_data_buffer, 
            lidar_data_lock, 
            camera_data_buffer, 
            camera_data_lock, 
            attached_sensors, 
            ego_vehicle, proximity_mapping
        )
        vehicle_mapping[vehicle_label]["sensors"] = smart_sensors
        logging.info(f"{vehicle_label} has {len(smart_sensors)} sensors attached.")
    vehicle_mapping["ego_veh"]["sensors"] = ego_vehicle_sensors

    # Save vehicle mapping to a JSON file
    save_vehicle_mapping(vehicle_mapping)

    # Initialize WaypointManager with radius-based region
    region_center = carla.Location(x=0, y=0, z=0)  # Define the center of the operational region
    region_radius = 50.0  # 50 meters radius
    waypoint_manager = WaypointManager(
        world=world,
        vehicle_mapping=vehicle_mapping,
        region_center=region_center,
        region_radius=region_radius
    )


    # Generate initial waypoints for smart vehicles
    waypoint_manager.generate_initial_waypoints()

    #Spawn NPC vehicles and walkers
    traffic_manager.set_global_distance_to_leading_vehicle(config.simulation.npc_global_dist_lv)
    traffic_manager.set_hybrid_physics_mode(True)#Only works if we have vehicle tagged with role_name = 'hero'
    traffic_manager.set_hybrid_physics_radius(70.0)#Need previous one to work first
    traffic_manager.set_respawn_dormant_vehicles(False)

    npc_vehicles = spawn_vehicles(client,world,traffic_manager, config.simulation.npc_num_vehicles)
    npc_walkers,npc_walker_speeds = spawn_walkers(client,world, config.simulation.npc_num_walkers)
    logging.info(f"Spawned {len(npc_vehicles)} NPC vehicles and {len(npc_walkers)} walkers.")

    # Attach a camera for visualization
    camera_transform = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera, camera_bp = attach_follow_camera(world, ego_vehicle, camera_transform)
    game_display, width, height = initialize_pygame(None, len(vehicle_mapping), fixed_resolution=(1920, 1080))
    render_object = RenderObject(width, height - 30)
    control_object = ControlObject(ego_vehicle)

    # Start camera feed
    camera.listen(lambda image: pygame_callback(image, render_object))

    try:
        game_loop(
            world, game_display, camera, render_object, control_object,
            vehicle_mapping, env_manager, ego_vehicle, smart_vehicles,
            lidar_data_buffer, lidar_data_lock, camera_data_buffer, camera_data_lock,
            waypoint_manager, birdview_producer, data_fusion_obs_det
        )

        logging.info("Bounding boxes plotted on all frames.")
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
    finally:
        logging.info("Shutting down simulation...")
        pygame.quit()  # Close PyGame window

        # Cleanup attached sensors
        logging.info("Cleaning up sensors...")
        for sensor in attached_sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
                    logging.info(f"Destroyed sensor ID: {sensor.id}")
            except Exception as e:
                logging.error(f"Failed to clean up sensor ID: {sensor.id} - {e}")
        logging.info("All sensors cleaned up successfully.")

        # Stop the ego listener thread
        ego_listener.stop_listener()
        ego_listener_thread.join()

        # Cleanup vehicles and other actors
        cleanup(client, vehicles, [])
        logging.info("Simulation terminated.")

if __name__ == "__main__":
    main()