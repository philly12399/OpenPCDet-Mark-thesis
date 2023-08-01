import pickle
import argparse
import os.path as osp
import os
import sys
import json
import numpy as np
import itertools
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velodyne_dir", type=str)
    parser.add_argument("--pcd_dir", type=str)
    return parser.parse_args()

def read_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[11:]:
            linestr = line.split(" ")
            linestr_convert = list(map(float, linestr[:4]))
            lidar.append(linestr_convert)
    return np.array(lidar)

if __name__ == '__main__':
    args = parse_args()
    pcd_dir = args.pcd_dir
    velodyne_dir = args.velodyne_dir
    if velodyne_dir is not None:
        files = os.listdir(velodyne_dir)
        files.sort()
        for file in tqdm(files):
            points = np.fromfile(osp.join(velodyne_dir, file),
                                dtype=np.float32).reshape(-1, 4)
            ratio_map = defaultdict(int)
            for point in points:
                ratio_map[int(point[3] / 255. * 10.)] += 1
            for k in ratio_map.keys():
                ratio_map[k] = f'{ratio_map[k] / len(points):.2}'
            print(ratio_map)
    elif pcd_dir is not None:
        files = os.listdir(pcd_dir)
        files.sort()
        for file in tqdm(files):
            points = read_pcd(osp.join(pcd_dir, file))
            max_intensity = np.max(points, axis=0)[3]
            min_intensity = np.min(points, axis=0)[3]
            assert max_intensity <= 255
            assert min_intensity >=0
            # ratio_map = defaultdict(int)
            # for point in points:
            #     ratio_map[int(point[3] / 255. * 10.)] += 1
            # for k in ratio_map.keys():
            #     ratio_map[k] = f'{ratio_map[k] / len(points):.2}'
            # print(ratio_map)

