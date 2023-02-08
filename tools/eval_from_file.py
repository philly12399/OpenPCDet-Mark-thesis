import _init_path
import argparse
import datetime
import glob
import os
import re
import time
import copy

from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.models import build_network
from pcdet.utils import common_utils, object3d_kitti, calibration_kitti


def get_annos_from_kitti_dir(kitti_dir, idx):
    label_path = kitti_dir / 'label_2' / ('%s.txt' % idx)
    calib_path = kitti_dir / 'calib' / ('%s.txt' % idx)
    obj_list = object3d_kitti.get_objects_from_label(label_path)

    calib = calibration_kitti.Calibration(calib_path)
    annotations = {}
    annotations['name'] = np.array(
        [obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array(
        [obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array(
        [obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array(
        [obj.alpha for obj in obj_list])
    if len(obj_list) != 0:
        annotations['bbox'] = np.concatenate(
            [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    else:
        annotations['bbox'] = np.zeros((0, 4))
    annotations['dimensions'] = np.array(
        [[obj.l, obj.h, obj.w] for obj in obj_list]).reshape(-1, 3)  # lhw(camera) format
    if len(obj_list) != 0:
        annotations['location'] = np.concatenate(
            [obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    else:
        annotations['location'] = np.zeros((0, 3))
    annotations['rotation_y'] = np.array(
        [obj.ry for obj in obj_list])
    annotations['score'] = np.array(
        [obj.score for obj in obj_list])
    annotations['difficulty'] = np.array(
        [obj.level for obj in obj_list], np.int32)

    num_objects = len(
        [obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + \
        [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)

    loc = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate(
        [loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default=None,
    #                     help='specify the config for training')
    # parser.add_argument(
    #     '--eval_range', action='store_true', default=False, help='')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='specify the kitti directory storing groundtruth labels')
    parser.add_argument('--det_dir', type=str, default=None,
                        help='specify the kitti directory storing predicted labels')

    args = parser.parse_args()

    # cfg_from_yaml_file(args.cfg_file, cfg)
    # cfg.TAG = Path(args.cfg_file).stem
    # cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    # np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()

    gt_dir = Path(args.gt_dir)
    gt_indices = [x.split('.')[0]
                  for x in os.listdir(gt_dir / 'label_2')]
    gt_indices.sort()

    det_dir = Path(args.det_dir)

    gt_annos = []
    det_annos = []

    for idx in gt_indices:
        gt_anno = get_annos_from_kitti_dir(gt_dir, idx)
        det_anno = get_annos_from_kitti_dir(det_dir, idx)
        # Set score to random
        # for i in range(len(det_anno['score'])):
        #     det_anno['score'][i] = 1.
        # Transform the annos a little bit
        # for i in range(len(annos['location'])):
        #     offset = 0.01
        #     annos['location'][i][0] += random.uniform(-offset, offset)
        #     annos['location'][i][2] += random.uniform(-offset, offset)
        if len(gt_anno['name']) != 0:
            gt_annos += [gt_anno]
            det_annos += [det_anno]

    ranges = (0, 30, 50, 80)
    result_str, dict = kitti_eval.get_range_eval_result(
        gt_annos, det_annos, ['Cyclist', 'Car', 'Truck', 'Bus'], ranges=ranges)
    print(result_str)

    # # create logger
    # output_dir = cfg.ROOT_DIR / 'output' / \
    #     cfg.EXP_GROUP_PATH / cfg.TAG
    # output_dir.mkdir(parents=True, exist_ok=True)

    # eval_output_dir = output_dir / 'eval' / 'eval_all_default'

    # eval_output_dir.mkdir(parents=True, exist_ok=True)
    # log_file = eval_output_dir / \
    #     ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # # log to file
    # logger.info('**********************Start logging**********************')
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(
    # ) else 'ALL'
    # logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # for key, val in vars(args).items():
    #     logger.info('{:16} {}'.format(key, val))
    # log_config_to_file(cfg, logger=logger)

    # gt_set, gt_loader, gt_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=2, dist=False, workers=4, logger=logger, training=False
    # )

    # test_dataset_cfg = copy.deepcopy(cfg.DATA_CONFIG)
    # test_dataset_cfg['DATA_PATH'] = args.kitti_test_dir

    # test_set, test_loader, test_sampler = build_dataloader(
    #     dataset_cfg=test_dataset_cfg,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=2, dist=False, workers=4, logger=logger, training=False
    # )

    # eval_utils.eval_from_dataset(
    #     cfg, test_loader, gt_loader, 0, logger,
    #     result_dir=eval_output_dir, save_to_file=False, eval_range=args.eval_range
    # )


if __name__ == '__main__':
    main()
