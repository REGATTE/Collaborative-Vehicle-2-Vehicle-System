import carla
import logging
import random
import numpy as np
import time

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

"""

class WaypointManager:
    def __init__(self, world, vehicle_mapping, region_center, region_radius):
        """
        Initializes the Waypoint Manager with a circular operational region.
        :param world: CARLA world instance.
        :param vehicle_mapping: Vehicle mapping dictionary with initial positions.
        :param region_center: Center of the operational region as a carla.Location object.
        :param region_radius: Radius of the operational region in meters.
        :param proximity_radius: Radius for proximity-based interactions.
        """
        self.world = world
        self.map = world.get_map()
        self.vehicle_mapping = vehicle_mapping
        self.region_center = region_center
        self.region_radius = region_radius
        self.waypoints = {}
        self.update_counter = 0
        self.last_warning_time = {}  # Track last warning log time per vehicle

    def generate_initial_waypoints(self, num_waypoints=10, spacing=10.0):
        """
        Generates initial waypoints for all vehicles (excluding the ego vehicle) within the defined circular region.
        """
        for label, data in self.vehicle_mapping.items():
            if label == "ego_veh":
                continue  # Skip generating waypoints for the ego vehicle

            initial_position = data["initial_position"]
            base_location = carla.Location(
                x=initial_position["x"], y=initial_position["y"], z=initial_position["z"]
            )
            # Ensure each vehicle has a consistent dictionary structure
            self.waypoints[label] = {
                "waypoints": self._generate_waypoints(base_location, num_waypoints, spacing),
                "logged_diversion": False
            }
            logging.info(f"Generated {num_waypoints} waypoints for {label} starting from {initial_position}.")

    def _generate_waypoints(self, base_location, num_waypoints, spacing):
        """
        Generates waypoints from a base location while ensuring they stay within the circular region.
        """
        waypoints = []
        for i in range(num_waypoints):
            offset_location = base_location + carla.Location(x=spacing * i, y=0, z=0)
            if self._is_within_region(offset_location):
                waypoints.append(self.map.get_waypoint(offset_location))
            else:
                break  # Stop generating waypoints if out of the circular region
        return waypoints

    def _is_within_region(self, location):
        """
        Checks if a given location is within the defined circular region.
        """
        distance_to_center = location.distance(self.region_center)
        return distance_to_center <= self.region_radius

    def update_waypoints_for_vehicle(self, vehicle_label, target_location=None, num_waypoints=5, spacing=10.0):
        """
        Updates waypoints for a specific vehicle.
        If a target_location is provided, generate waypoints leading to it.
        Diverts smart vehicles to independent paths if they are too close to the ego vehicle.
        """
        if vehicle_label not in self.waypoints:
            self.waypoints[vehicle_label] = {"waypoints": [], "logged_diversion": False}

        vehicle_data = self.vehicle_mapping[vehicle_label]
        actor = self.world.get_actor(vehicle_data["actor_id"])
        if not actor:
            return

        current_location = actor.get_transform().location

        if target_location:
            distance_to_target = current_location.distance(target_location)
            if distance_to_target <= self.region_radius / 2:
                if not self.waypoints[vehicle_label]["logged_diversion"]:
                    logging.info(f"Vehicle {vehicle_label} is too close to the ego vehicle. Diverting to an independent path.")
                    self.waypoints[vehicle_label]["logged_diversion"] = True

                self.waypoints[vehicle_label]["waypoints"] = self._generate_waypoints(
                    current_location, num_waypoints, spacing
                )
                return

            direction_vector = target_location - current_location
            normalized_direction = direction_vector / max(direction_vector.length(), 1e-6)  # Avoid division by zero
            self.waypoints[vehicle_label]["waypoints"] = [
                self.map.get_waypoint(current_location + normalized_direction * spacing * i)
                for i in range(num_waypoints)
            ]
            #logging.debug(f"Updated waypoints for {vehicle_label} to move toward target location.")
        else:
            self.waypoints[vehicle_label]["waypoints"] = self._generate_waypoints(
                current_location, num_waypoints, spacing
            )

    def manage_all_vehicles(self, periodic_update_interval=10):
        """
        Manages waypoint updates and assignments for all vehicles.
        Periodically directs smart vehicles toward the ego vehicle, with proximity checks for diversion.
        """
        # Get the ego vehicle's current location
        ego_data = self.vehicle_mapping["ego_veh"]
        ego_actor = self.world.get_actor(ego_data["actor_id"])
        ego_location = ego_actor.get_transform().location if ego_actor else None

        for vehicle_label in self.vehicle_mapping:
            if vehicle_label == "ego_veh":
                # Skip updating the ego vehicle's waypoints
                continue

            # Periodically update waypoints to converge to the ego vehicle
            if self.update_counter % periodic_update_interval == 0 and ego_location:
                self.update_waypoints_for_vehicle(vehicle_label, target_location=ego_location)
            else:
                self.assign_next_waypoint(vehicle_label)

        self.update_counter += 1

    def assign_next_waypoint(self, vehicle_label):
        """
        Assigns the next waypoint to the specified vehicle.
        Smoothly moves the vehicle toward the next waypoint.
        """
        if vehicle_label == "ego_veh":
            return  # Skip assigning waypoints for the ego vehicle

        if vehicle_label not in self.waypoints or not self.waypoints[vehicle_label]["waypoints"]:
            current_time = time.time()
            if (
                vehicle_label not in self.last_warning_time
                or current_time - self.last_warning_time[vehicle_label] > 2
            ):
                logging.warning(f"No waypoints available for {vehicle_label}. Generating new waypoints.")
                self.last_warning_time[vehicle_label] = current_time

            vehicle_data = self.vehicle_mapping[vehicle_label]
            actor = self.world.get_actor(vehicle_data["actor_id"])
            if actor:
                current_location = actor.get_transform().location
                self.waypoints[vehicle_label]["waypoints"] = self._generate_waypoints(
                    current_location, num_waypoints=5, spacing=10.0
                )
            return

        next_waypoint = self.waypoints[vehicle_label]["waypoints"].pop(0)
        vehicle_data = self.vehicle_mapping[vehicle_label]
        actor = self.world.get_actor(vehicle_data["actor_id"])

        if not actor:
            logging.warning(f"Vehicle {vehicle_label} with ID {vehicle_data['actor_id']} not found.")
            return

        target_location = next_waypoint.transform.location
        direction_vector = target_location - actor.get_transform().location
        normalized_direction = direction_vector / max(direction_vector.length(), 1e-6)  # Avoid division by zero
        velocity = normalized_direction * 5.0  # Adjust speed as needed
        actor.set_target_velocity(carla.Vector3D(x=velocity.x, y=velocity.y, z=velocity.z))

        self.update_counter += 1
        if self.update_counter % 5 == 0:
            logging.info(f"Assigned waypoint to {vehicle_label}.")