import carla
import logging
import random
import numpy as np
import math


"""
This code ensures that vehicles dynamically adjust their paths when they stray too far 
from their intended area of operation or predefined route.

The waypoints generated do not affect the movement of the ego vehicle.
---
Implementation Strategy

    Operational Region:
        Define a bounding region where both the ego and smart vehicles can operate. This ensures all vehicles stay 
        within a predefined area. Use this region as a constraint during waypoint generation. 
        The predefined region is set to 50m radius from the spawn point.

    Periodic Convergence:
        Introduce a timer or counter that triggers smart vehicles to generate new waypoints leading 
        toward the ego vehicle at regular intervals.

    Independent Paths:
        Between periodic updates, smart vehicles continue to follow their own independent waypoints.
---
To avoid all smart_vehicles converging on to the ego_vehicle after some time, the code has been updated to 

Generate Independent Waypoints:

    Redirect the smart vehicle to move away from the ego vehicle along an independent path.

Proximity Trigger:

    Add a check to stop generating waypoints toward the ego vehicle if the smart vehicle is within a certain range (e.g., proximity_radius / 2).
---

Added WaypointGenerator class to 
    Simplified State Management: 
        The generator does not maintain internal state for waypoints, reducing complexity.

    Dynamic Generation: 
        Waypoints are generated dynamically based on the current location and optional target location.

    Flexibility: 
        Supports both directed movement and randomized waypoint generation within a defined region.
"""

class WaypointGenerator:
    """
    A stateless generator for dynamic waypoint generation.
    """
    def __init__(self, world, region_center, region_radius):
        """
        Initializes the Waypoint Generator with a circular operational region.
        :param world: CARLA world instance.
        :param region_center: Center of the operational region as a carla.Location object.
        :param region_radius: Radius of the operational region in meters.
        """
        self.world = world
        self.map = world.get_map()
        self.center = region_center
        self.radius = region_radius

    def get_next_waypoint(self, current_location, target_location=None):
        """
        Generates the next waypoint based on the target location or circular path logic.
        :param current_location: Current location of the vehicle.
        :param target_location: Optional target location for directed movement.
        :return: The next waypoint.
        """
        if target_location and current_location.distance(target_location) > self.radius / 2:
            # Generate a waypoint moving toward the target
            return self._get_waypoint_toward(current_location, target_location)
        else:
            # Generate a waypoint within a circular path
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.3 * self.radius, 0.8 * self.radius)
            next_point = carla.Location(
                x=self.center.x + distance * math.cos(angle),
                y=self.center.y + distance * math.sin(angle),
                z=self.center.z
            )
            return self.map.get_waypoint(next_point)

    def _get_waypoint_toward(self, current_location, target_location):
        """
        Generates a waypoint moving toward a target location.
        :param current_location: Current location of the vehicle.
        :param target_location: Target location to move toward.
        :return: The next waypoint.
        """
        direction_vector = target_location - current_location
        normalized_direction = direction_vector / max(direction_vector.length(), 1e-6)  # Avoid division by zero
        next_location = current_location + normalized_direction * min(10.0, direction_vector.length() * 0.5)
        return self.map.get_waypoint(next_location)


class WaypointManager:
    """
    Manages vehicle waypoints using the stateless WaypointGenerator.
    """
    def __init__(self, world, vehicle_mapping, region_center, region_radius):
        """
        Initializes the Waypoint Manager with a circular operational region.
        :param world: CARLA world instance.
        :param vehicle_mapping: Vehicle mapping dictionary with initial positions.
        :param region_center: Center of the operational region as a carla.Location object.
        :param region_radius: Radius of the operational region in meters.
        """
        self.world = world
        self.vehicle_mapping = vehicle_mapping
        self.generator = WaypointGenerator(world, region_center, region_radius)
        self.update_counter = 0

    def update_waypoints_for_vehicle(self, vehicle_label, target_location=None):
        """
        Updates waypoints for a specific vehicle.
        If a target_location is provided, generate waypoints leading to it.
        :param vehicle_label: Label of the vehicle to update.
        :param target_location: Optional target location for directed movement.
        """
        vehicle_data = self.vehicle_mapping[vehicle_label]
        actor = self.world.get_actor(vehicle_data["actor_id"])
        if not actor:
            logging.warning(f"Vehicle {vehicle_label} not found.")
            return

        current_location = actor.get_transform().location
        next_waypoint = self.generator.get_next_waypoint(current_location, target_location)

        if next_waypoint:
            target_location = next_waypoint.transform.location
            direction_vector = target_location - current_location
            normalized_direction = direction_vector / max(direction_vector.length(), 1e-6)
            velocity = normalized_direction * 5.0  # Adjust speed as needed
            actor.set_target_velocity(carla.Vector3D(x=velocity.x, y=velocity.y, z=velocity.z))
            logging.info(f"Assigned new waypoint for {vehicle_label}.")

    def manage_all_vehicles(self, periodic_update_interval=10):
        """
        Manages waypoint updates and assignments for all vehicles.
        Periodically directs smart vehicles toward the ego vehicle, with proximity checks for diversion.
        """
        ego_data = self.vehicle_mapping["ego_veh"]
        if not ego_data:
            logging.error("Ego vehicle is not found in the vehicle mapping.")
            return  # Stop further processing
        ego_actor = self.world.get_actor(ego_data["actor_id"])
        if not ego_actor:
            logging.error("Ego vehicle actor could not be retrieved from the CARLA world.")
            return  # Stop further processing
        ego_location = ego_actor.get_transform().location if ego_actor else None
        if not ego_location:
            logging.error("Ego vehicle location is None. Skipping waypoint management for this iteration.")
            return  # Stop further processing
        
        for vehicle_label in self.vehicle_mapping:
            if vehicle_label == "ego_veh":
                continue

            if self.update_counter % periodic_update_interval == 0 and ego_location:
                self.update_waypoints_for_vehicle(vehicle_label, target_location=ego_location)
            else:
                self.update_waypoints_for_vehicle(vehicle_label)

        self.update_counter += 1