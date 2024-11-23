import carla
import logging
import random
import time

def setup_traffic_manager(client, port, hybrid_mode=True, respawn=False, seed=None):
    """
    Configure and return the Traffic Manager instance.
    
    Args:
        client (carla.Client): The CARLA client instance.
        port (int): Port number for the Traffic Manager.
        hybrid_mode (bool): Enable hybrid physics for smoother vehicle movements.
        respawn (bool): Enable respawning of dormant vehicles.
        seed (int, optional): Seed for randomization in Traffic Manager.
    
    Returns:
        traffic_manager (carla.TrafficManager): Configured Traffic Manager instance.
    """
    traffic_manager = client.get_trafficmanager(port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(hybrid_mode)
    traffic_manager.set_hybrid_physics_radius(70.0)
    traffic_manager.set_respawn_dormant_vehicles(respawn)
    if seed is not None:
        traffic_manager.set_random_device_seed(seed)
    return traffic_manager


def spawn_vehicles(client, world, traffic_manager, number_of_vehicles=10, safe_mode=True, vehicle_filter="vehicle.*", vehicle_generation="All"):
    """
    Spawn vehicles in the simulation with specified configurations.
    Ensures only cars are spawned if safe_mode is True.
    """
    # Define ego vehicle and smart vehicle blueprint filters
    ego_vehicle_blueprint_id = "vehicle.mercedes.coupe_2020"
    smart_vehicle_blueprints = [
        "vehicle.dodge.charger_2020",
        "vehicle.dodge.police",
        "vehicle.ford.crown",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s",
        "vehicle.nissan.patrol"
    ]

    # Get all vehicle blueprints
    all_blueprints = get_actor_blueprints(world, vehicle_filter, vehicle_generation)

    # Get the blueprint for the ego vehicle
    ego_vehicle_blueprint = next(
        (bp for bp in all_blueprints if bp.id == ego_vehicle_blueprint_id), None
    )

    # Filter blueprints for smart vehicles
    smart_vehicle_blueprints = [
        bp for bp in all_blueprints if bp.id in smart_vehicle_blueprints
    ]

    if not ego_vehicle_blueprint:
        logging.error("Ego vehicle blueprint not found!")
        return []

    if not smart_vehicle_blueprints:
        logging.error("No smart vehicle blueprints found!")
        return []

    # Shuffle spawn points
    spawn_points = world.get_map().get_spawn_points()
    logging.info(f"Available spawn points: {len(spawn_points)}")
    random.shuffle(spawn_points)

    if len(spawn_points) < number_of_vehicles:
        logging.warning("Not enough spawn points for the requested number of vehicles.")
        number_of_vehicles = len(spawn_points)

    vehicles = []
    batch = []

    # Spawn the ego vehicle
    logging.info("Spawning the ego vehicle...")
    ego_spawn_point = spawn_points.pop(0)
    ego_vehicle_blueprint.set_attribute('role_name', 'ego_vehicle')
    batch.append(
        carla.command.SpawnActor(ego_vehicle_blueprint, ego_spawn_point)
        .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port()))
    )

    # Spawn the smart vehicles
    logging.info("Spawning smart vehicles...")
    for i, transform in enumerate(spawn_points[:number_of_vehicles - 1]):
        blueprint = random.choice(smart_vehicle_blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', f'smart_vehicle_{i + 1}')
        batch.append(
            carla.command.SpawnActor(blueprint, transform)
            .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port()))
        )

    # Apply the batch
    responses = client.apply_batch_sync(batch, True)
    for response in responses:
        if response.error:
            logging.error(f"Error spawning vehicle: {response.error}")
        else:
            vehicles.append(response.actor_id)

    logging.info("Vehicles successfully spawned.")
    return vehicles

def spawn_walkers(client, world, number_of_walkers=20, walker_filter="walker.pedestrian.*", walker_generation="2", seed=None):
    """
    Spawn pedestrians (walkers) in the simulation.
    
    Args:
        client (carla.Client): The CARLA client instance.
        world (carla.World): The CARLA world instance.
        number_of_walkers (int): Number of walkers to spawn.
        walker_filter (str): Blueprint filter pattern for walkers.
        walker_generation (str): Walker generation to filter (e.g., "2").
        seed (int, optional): Seed for randomization.
    
    Returns:
        walkers (list): List of spawned walker actor IDs.
        walker_speeds (list): List of assigned walking speeds.
    """
    walker_list = []
    all_id = []
    walker_speeds = []
    percentage_pedestrians_crossing = 0.2
    percentage_pedestrians_running = 0.2
    
    # Set random seed if provided
    if seed is not None:
        world.set_pedestrians_seed(seed)
        random.seed(seed)
    
    # Get walker blueprints
    walker_blueprints = get_actor_blueprints(world, walker_filter, walker_generation)
    
    # 1. Get spawn points
    spawn_points = []
    for _ in range(number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    
    # 2. Spawn walker actors
    batch = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(walker_blueprints)
        
        # Set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        
        # Set speed
        if walker_bp.has_attribute('speed'):
            if random.random() > percentage_pedestrians_running:
                # walking
                walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            logging.warning("Walker has no speed attribute")
            walker_speeds.append(0.0)
            
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
    
    # Apply the batch and create walkers
    results = client.apply_batch_sync(batch, True)
    
    # Process results and update speed list
    walker_speeds_filtered = []
    for i, result in enumerate(results):
        if result.error:
            logging.error(result.error)
        else:
            walker_list.append({"id": result.actor_id})
            walker_speeds_filtered.append(walker_speeds[i])
    walker_speeds = walker_speeds_filtered
    
    # 3. Spawn walker controllers
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walker_list:
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))
    
    # Apply the batch and create controllers
    results = client.apply_batch_sync(batch, True)
    for i, result in enumerate(results):
        if result.error:
            logging.error(result.error)
        else:
            walker_list[i]["con"] = result.actor_id
    
    # 4. Create id list for all walkers and controllers
    for walker in walker_list:
        all_id.append(walker["con"])
        all_id.append(walker["id"])
    all_actors = world.get_actors(all_id)
    
    # Wait for a tick to ensure client receives the last transform
    world.tick()
    
    # 5. Initialize controllers and set walking behavior
    world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)
    for i in range(0, len(all_id), 2):
        # Start walker
        all_actors[i].start()
        # Set random destination
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # Set max speed
        all_actors[i].set_max_speed(float(walker_speeds[int(i/2)]))
    
    return walker_list, walker_speeds


def get_actor_blueprints(world, filter_pattern, generation):
    """
    Retrieve actor blueprints filtered by type and generation.
    
    Args:
        world (carla.World): The CARLA world instance.
        filter_pattern (str): Blueprint filter pattern.
        generation (str): Generation to filter (e.g., "2" or "All").
    
    Returns:
        list: Filtered list of blueprints.
    """
    blueprints = world.get_blueprint_library().filter(filter_pattern)
    if generation.lower() == "all":
        return blueprints
    try:
        gen = int(generation)
        return [bp for bp in blueprints if int(bp.get_attribute('generation')) == gen]
    except:
        logging.warning("Invalid generation filter. Returning all blueprints.")
        return blueprints


def cleanup(client, vehicles, walkers):
    """
    Destroy all spawned actors (vehicles and walkers).
    
    Args:
        client (carla.Client): The CARLA client instance.
        vehicles (list): List of vehicle actor IDs.
        walkers (list): List of walker actor IDs.
    """
    if vehicles:
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
    if walkers:
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers])


# Example integration with a script
def main():
    """
    Main function demonstrating the setup and cleanup of traffic simulation.
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    traffic_manager = setup_traffic_manager(client, port=8000, hybrid_mode=True, respawn=True, seed=42)

    try:
        # Spawn vehicles
        vehicles = spawn_vehicles(client, world, traffic_manager, number_of_vehicles=20)

        # Spawn walkers
        walkers, walker_speeds = spawn_walkers(client, world, number_of_walkers=15)

        logging.info(f"Spawned {len(vehicles)} vehicles and {len(walkers)} walkers.")
        while True:
            world.tick()
    finally:
        cleanup(client, vehicles, walkers)
        logging.info("Cleaned up all actors.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Simulation interrupted.")
    finally:
        logging.info("Simulation ended.")
