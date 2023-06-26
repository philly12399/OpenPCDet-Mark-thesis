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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    kitti_dir = args.kitti_dir
    output_dir = osp.join(kitti_dir, '../unified_kitti_format')
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(osp.join(output_dir, 'label_2'))
    os.symlink(osp.join(kitti_dir, 'calib'), osp.join(output_dir, 'calib'))
    os.symlink(osp.join(kitti_dir, 'velodyne'),
               osp.join(output_dir, 'velodyne'))
    label_dir = osp.join(kitti_dir, 'label_2')
    files = os.listdir(label_dir)
    files.sort()

    for file in tqdm(files):
        file_path = osp.join(label_dir, file)
        lines = open(file_path, 'r').readlines()
        output_str = ''
        for line in lines:
            arr = line.split(' ')
            arr[0] = 'Dynamic'
            output_str = output_str + ' '.join(arr)
        output_path = osp.join(output_dir, 'label_2', file)
        with open(output_path, 'w') as f:
            f.write(output_str)
