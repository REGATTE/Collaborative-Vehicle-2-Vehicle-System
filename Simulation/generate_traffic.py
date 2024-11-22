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
    
    Args:
        client (carla.Client): The CARLA client instance.
        world (carla.World): The CARLA world instance.
        traffic_manager (carla.TrafficManager): The configured Traffic Manager.
        number_of_vehicles (int): Number of vehicles to spawn.
        safe_mode (bool): Spawn only cars if True.
        vehicle_filter (str): Blueprint filter pattern for vehicles.
        vehicle_generation (str): Vehicle generation to filter (e.g., "2" or "All").
    
    Returns:
        vehicles (list): List of spawned vehicle actor IDs.
    """
    vehicle_blueprints = get_actor_blueprints(world, vehicle_filter, vehicle_generation)
    if safe_mode:
        vehicle_blueprints = [bp for bp in vehicle_blueprints if bp.get_attribute('base_type') == 'car']

    spawn_points = world.get_map().get_spawn_points()
    logging.info(f"Available spawn points before filtering: {len(spawn_points)}")
    random.shuffle(spawn_points)
    number_of_vehicles = min(number_of_vehicles, len(spawn_points))

    vehicles = []
    batch = []
    for i, transform in enumerate(spawn_points[:number_of_vehicles]):
        blueprint = random.choice(vehicle_blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(blueprint, transform)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

    responses = client.apply_batch_sync(batch, True)
    for response in responses:
        if response.error:
            logging.error(response.error)
        else:
            vehicles.append(response.actor_id)

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
    walker_blueprints = get_actor_blueprints(world, walker_filter, walker_generation)
    spawn_points = []
    for _ in range(number_of_walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))

    walkers = []
    walker_speeds = []
    batch = []
    for spawn_point in spawn_points:
        blueprint = random.choice(walker_blueprints)
        if blueprint.has_attribute('speed'):
            walker_speeds.append(random.choice(blueprint.get_attribute('speed').recommended_values))
        else:
            walker_speeds.append(1.4)
        batch.append(carla.command.SpawnActor(blueprint, spawn_point))

    walker_responses = client.apply_batch_sync(batch, True)
    for response in walker_responses:
        if response.error:
            logging.error(response.error)
        else:
            walkers.append(response.actor_id)

    return walkers, walker_speeds


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
