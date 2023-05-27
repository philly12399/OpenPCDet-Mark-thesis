import fire
import _init_path
# import yaml
from ruamel import yaml
# import ruamel.yaml

# from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils


def add_cfgs(data_path):
    kitti_cfg_path = 'cfgs/kitti_models/pointrcnn_wayside.yaml'
    kitti_cfg = yaml.safe_load(open(kitti_cfg_path, 'r'))

    dataset_cfg = yaml.safe_load(
        open(kitti_cfg['DATA_CONFIG']['_BASE_CONFIG_'], 'r'))
    dataset_cfg['DATA_PATH'] = data_path
    dataset_name = data_path.split('/')[-1]
    new_dataset_path = f'cfgs/dataset_configs/kitti_{dataset_name}.yaml'
    kitti_cfg['DATA_CONFIG']['_BASE_CONFIG_'] = new_dataset_path
    new_kitti_path = f'cfgs/kitti_models/pointrcnn_{dataset_name}.yaml'

    with open(new_kitti_path, 'w') as f:
        yaml.safe_dump(kitti_cfg, f, indent=4)
        print(f'add pointrcnn config to {new_kitti_path}')

    with open(new_dataset_path, 'w') as f:
        yaml.safe_dump(dataset_cfg, f, indent=4)
        print(f'add dataset config to {new_dataset_path}')


if __name__ == '__main__':
    fire.Fire()
