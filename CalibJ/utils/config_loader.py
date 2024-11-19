import yaml
import os

from types import SimpleNamespace

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
