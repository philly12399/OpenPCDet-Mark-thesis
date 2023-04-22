import _init_path
import argparse
import numpy as np
import os
import re

from pathlib import Path

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None,
                        required=False, help='batch size for training')
    parser.add_argument('--self_training_dir', type=str, default=None,
                        help='specify a self-training directory to be evaluated if needed')
    parser.add_argument('--rounds', type=str, default=None,
                        help='specify a start-end rounds to evaluate')
    parser.add_argument(
        '--save_to_file', action='store_true', default=False, help='')
    parser.add_argument(
        '--eval_range', action='store_false', default=True, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()
    st_dirs = [x for x in os.listdir(
        args.self_training_dir) if x.startswith('pv_rcnn_ST-lidar1-r')]
    st_dirs.sort(key=lambda x: int(x.split('r')[-1]))

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, training=False
    )

    logger = common_utils.create_logger()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=test_set)

    result_types = ['Cyclist_bev_0.25', 'Cyclist_3d_0.25',
                    'Cyclist_bev_0.3', 'Cyclist_3d_0.3',
                    'Car/Truck_bev_0.5', 'Car/Truck_3d_0.5',
                    'Car/Truck_bev_0.7', 'Car/Truck_3d_0.7']
    result_types = [x.ljust(20) for x in result_types]
    result_str = ","+",".join(result_types)+"\n"
    
    for st_dir in st_dirs:
        round_id = int(st_dir.split('r')[-1])
        if args.rounds is not None:
            start = int(args.rounds.split('-')[0])
            end = int(args.rounds.split('-')[1])
            if round_id < start or round_id > end:
                continue
        output_dir = cfg.ROOT_DIR / 'output' / \
            cfg.EXP_GROUP_PATH / cfg.TAG / ('ST-R%s' % round_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_output_dir = output_dir / 'eval' / 'eval_all_default'

        eval_output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = os.path.join(args.self_training_dir, st_dir,
                                 'default/ckpt/checkpoint_epoch_80.pth')
        print(ckpt_path)
        # load checkpoint
        model.load_params_from_file(
            filename=ckpt_path, logger=logger)
        model.cuda()

        return_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, 80, logger, args, result_dir=eval_output_dir,)

        new_str = f'R_{round_id},' + \
            f'{return_dict["Cyclist_bev_iou0.25/00-80_R40"]:.3f},' + \
            f'{return_dict["Cyclist_3d_iou0.25/00-80_R40"]:.3f},' + \
            f'{return_dict["Cyclist_bev_iou0.3/00-80_R40"]:.3f},' + \
            f'{return_dict["Cyclist_3d_iou0.3/00-80_R40"]:.3f},' + \
            f'{return_dict["Car/Truck_bev_iou0.5/00-80_R40"]:.3f},' + \
            f'{return_dict["Car/Truck_3d_iou0.5/00-80_R40"]:.3f},' + \
            f'{return_dict["Car/Truck_bev_iou0.7/00-80_R40"]:.3f},' + \
            f'{return_dict["Car/Truck_3d_iou0.7/00-80_R40"]:.3f},' + "\n"

        result_str += new_str
        print(result_str)

    output_dir = os.path.join(args.self_training_dir, 'self_training_results.csv')
    with open(output_dir, 'w') as f:
        f.write(result_str)
    print(result_str)


if __name__ == '__main__':
    main()
