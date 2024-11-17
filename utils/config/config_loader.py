import yaml
import logging


class DotDict(dict):
    """
    A dictionary that supports dot notation for nested access.
    """
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):  # Convert nested dictionaries to DotDict
            value = DotDict(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def load_config(config_path="config.yaml"):
    """
    Load configuration settings from a YAML file with dot-notation support.
    :param config_path: Path to the YAML configuration file.
    :return: DotDict containing configuration settings.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                logging.error(f"Configuration file does not contain a valid dictionary: {config_path}")
                raise ValueError(f"Configuration file does not contain a valid dictionary: {config_path}")
            logging.info(f"Configuration loaded successfully from {config_path}")
            return DotDict(config)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {config_path}\n{e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the configuration: {e}")
        raise
