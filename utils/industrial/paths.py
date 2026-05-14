from pathlib import Path
import os
from glob import glob
import json
from datetime import datetime


def get_root() -> Path:
    script_path = os.path.realpath(__file__)
    root_path = Path(script_path).parent.parent.parent
    return root_path

def get_dir(key: str) -> Path:
    config_value = get_root_config()[key]
    if os.path.isabs(config_value):
        return Path(config_value)
    else:
        return get_root() / config_value

def get_weights_path(model_name: str) -> Path:
    weights_path = glob(os.path.join(get_dir('dino_weights_dir'), f'{model_name}_pretrain_*.pth'))[0]
    return Path(weights_path)

def get_model_path(category: str) -> Path:
    models_dir = get_dir('models_dir')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir / category

def get_result_anomaly_images_path(category: str, split: str):
    result_dir = get_dir('results_dir') / 'anomaly_images' / category / split
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_results_path():
    result_dir = get_dir('results_dir')
    return result_dir

def get_result_anomaly_images_thresholded_path(category: str, split: str):
    result_dir = get_dir('results_dir') / 'anomaly_images_thresholded' / category / split
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_config_path():
    return get_root() / 'config.json'

def get_dataset_path(dataset_name: str):
    return get_dir('datasets_dir') / dataset_name

def get_submission_path():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'submission_{timestamp}.tar.gz'
    return get_dir('submissions_dir') / filename

def get_root_config():
    filepath = get_root() / 'config.json'
    return json.load(open(filepath, 'r'))

def get_result_csv_path(name: str):
    result_dir = get_dir('results_dir') / f'{name}.csv'
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    return result_dir
