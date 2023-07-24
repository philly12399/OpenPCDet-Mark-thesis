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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    velodyne_dir = args.velodyne_dir
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
