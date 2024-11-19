import yaml
import os
import json
from types import SimpleNamespace
import numpy as np

def load_config(config_file='config/calib_setting.yaml'):
    # ROS2 워크스페이스에서 src 디렉토리 경로 계산
    # workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    # src_dir = os.path.join(workspace_dir, 'src', 'CalibJ')
    # config_path = os.path.join(src_dir, config_file)
    config_path = "/home/f1tenth/kjy_ws/src/CalibJ/config/calib_setting.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Configuration file path: {config_path}")  # 디버깅용 출력

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return SimpleNamespace(**config)


def load_json(file_path):
    """
    Load camera calibration parameters from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        SimpleNamespace: An object containing camera_matrix and dist_coeffs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to SimpleNamespace for easy access
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32).flatten()

        return SimpleNamespace(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration JSON file not found: {file_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in JSON file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")