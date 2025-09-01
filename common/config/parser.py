import yaml
from typing import Any, Dict

class TrainingConfig:
    """
    A simple wrapper class for accessing configuration parameters.
    Allows accessing dictionary keys as attributes.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, TrainingConfig(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"TrainingConfig({self._config})"

    def get(self, key: str, default: Any = None) -> Any:
        """Provides a way to get a value with a default, similar to dict.get()."""
        return self._config.get(key, default)

def load_config(config_path: str) -> TrainingConfig:
    """
    Loads a YAML configuration file from the given path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        TrainingConfig: An object containing the configuration parameters.
    """
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig(config_dict)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{config_path}'")
        raise
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML file at '{config_path}': {e}")
        raise
