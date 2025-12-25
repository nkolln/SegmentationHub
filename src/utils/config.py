import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
