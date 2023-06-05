import pickle
import argparse
import os.path as osp
import os
import sys
import json
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--index_map", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    kitti_dir = args.kitti_dir
    output_dir = args.output_dir
    if os.path.isdir(output_dir):
        print(f"output folder {output_dir} already exists !!")
        sys.exit()

    output_folders = ['label_2', 'calib', 'velodyne', 'image_2']
    suffix_map = {
        'label_2': 'txt',
        'calib': 'txt',
        'velodyne': 'bin',
        'image_2': 'png',
    }

    for folder in output_folders:
        os.makedirs(osp.join(output_dir, folder))

    index_map = json.load(open(args.index_map, 'r'))

    for folder in output_folders:
        for wayside_idx, gt_idx in index_map.items():
            src_name = f'{int(gt_idx):06}.{suffix_map[folder]}'
            dst_name = f'{int(wayside_idx):06}.{suffix_map[folder]}'
            # src_name = ("%06d" % int(gt_idx)) + '.' + suffix_map[folder]
            # dst_name = ("%06d" % int(wayside_idx)) + '.' + suffix_map[folder]
            src_path = osp.join(osp.join(kitti_dir, folder), src_name)
            dst_path = osp.join(osp.join(output_dir, folder), dst_name)
            print(f'linking from {src_path} to {dst_path}.')
            os.symlink(src_path, dst_path)
