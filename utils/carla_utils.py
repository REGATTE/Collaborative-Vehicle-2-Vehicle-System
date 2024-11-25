import carla
import logging

def initialize_carla(host='localhost', port=2000, timeout=10.0):
    """
    Initializes the CARLA client and retrieves the world object.
    :param host: Hostname of the CARLA server.
    :param port: Port of the CARLA server.
    :param timeout: Timeout for the client connection.
    :return: Tuple of CARLA client and world objects.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    logging.info("Connected to CARLA server.")
    return client, world

def setup_synchronous_mode(world, traffic_manager, config):
    """
    Sets up CARLA's synchronous mode for deterministic simulation.
    :param world: The CARLA world object.
    :param traffic_manager: The CARLA traffic manager.
    :param config: Configuration object.
    """
    settings = world.get_settings()
    settings.synchronous_mode = config.simulation.synchronous_mode
    settings.fixed_delta_seconds = config.simulation.fixed_delta_seconds
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)
    logging.info("Synchronous mode enabled.")
