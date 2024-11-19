import json
import logging
import os

MAPPING_FILE_PATH = "utils/vehicle_mapping/ego_vehicle_mapping.json"

def save_vehicle_mapping(vehicle_mapping):
    """
    Saves the vehicle mapping to a JSON file, ensuring only serializable data is saved.
    :param vehicle_mapping: Dictionary containing vehicle and sensor mappings.
    """
    serializable_mapping = {}

    try:
        for label, data in vehicle_mapping.items():
            serializable_mapping[label] = {
                "actor_id": data["actor"].id,  # Save only the actor ID
                "sensors": [sensor.id for sensor in data["sensors"]],  # Save sensor IDs
            }

        with open(MAPPING_FILE_PATH, "w") as file:
            json.dump(serializable_mapping, file, indent=4)
            file.flush()
            os.fsync(file.fileno())  # Ensure data is written to disk
        logging.info(f"Vehicle mapping saved to {MAPPING_FILE_PATH}")
    except Exception as e:
        logging.error(f"Error saving vehicle mapping: {e}")

def load_vehicle_mapping(file_path):
    """
    Load the vehicle mapping from the given JSON file.
    :param file_path: Path to the JSON file containing vehicle mapping data.
    :return: Dictionary with vehicle mapping or None if the file cannot be loaded.
    """
    try:
        with open(file_path, 'r') as file:
            mapping = json.load(file)
        logging.info(f"Vehicle mapping loaded successfully from {file_path}.")
        return mapping
    except FileNotFoundError:
        logging.error(f"Vehicle mapping file not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding vehicle mapping JSON: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading vehicle mapping: {e}")
    return None