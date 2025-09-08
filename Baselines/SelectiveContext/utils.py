import os
import sys
import yaml
from typing import Dict, Any, Optional
from datasets import Dataset, load_from_disk


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_filename(config: Dict[str, Any]) -> str:
    """Generate standardized dataset filename from config"""
    filename = f"MS-MARCO-compressed-{config['compressor']['reduce_ratio']}-{config['compressor']['reduce_level']}"
    return filename.replace(".", "_")


def get_dataset_path(config: Dict[str, Any], filename_suffix: str = "") -> str:
    """Get full path for dataset storage"""
    filename = get_dataset_filename(config) + filename_suffix
    return os.path.join(config['save_dir'], filename)


def get_metadata_path(config: Dict[str, Any]) -> str:
    """Get path for metadata JSON file"""
    filename = get_dataset_filename(config)
    return os.path.join(config['save_dir'], f"{filename}.json")


def load_compressed_dataset(config: Dict[str, Any]) -> Dataset:
    """Load compressed dataset from results directory"""
    dataset_path = get_dataset_path(config)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Compressed dataset not found at: {dataset_path}")
    
    print(f"Loading compressed dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} samples")
    return dataset


def parse_config_arg() -> Optional[str]:
    """Parse command line argument for config file"""
    if len(sys.argv) > 2:
        return None  # Error case
    return sys.argv[1] if len(sys.argv) == 2 else "experiment.yaml"


def setup_experiment(script_name: str) -> Dict[str, Any]:
    """Common setup for all experiment scripts"""
    config_path = parse_config_arg()
    if config_path is None:
        print(f"Usage: python {script_name} [config_file.yaml]")
        sys.exit(1)
    
    try:
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    return config