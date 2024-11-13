import yaml
import os

class Config:
    """A class to allow dot-notation access to configuration parameters."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def load_config(path="config.yaml"):
    """Load the YAML configuration file."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file '{path}' not found.")
        
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        if not config_dict:
            raise ValueError("Configuration file is empty or invalid.")
        
        return Config(config_dict)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    
    except ValueError as e:
        print(f"Error: {e}")
        raise
