from os import path as osp
from tqdm import tqdm

import numpy as np
import argparse
import os
import json


def gen_default_roi(args):
    output_dir = args.output_dir
    wayside_groups = json.load(open(args.wayside_groups, 'r'))

    if osp.isdir(output_dir):
        print(f"output dir {output_dir} already exists.")
        return

    os.mkdir(output_dir)

    for wayside_id, indices in enumerate(tqdm(wayside_groups)):
        roi_struct = {
            'road_regions': [],
            'background_regions': [{
                'center_x': 0.,
                'center_y': 0.,
                'center_z': 0.,
                'size_x': 4.,
                'size_y': 2.,
                'size_z': 2.,
                'rot_z': 0.,
            }],
        }
        with open(osp.join(output_dir, f'{wayside_id:06}.json'), 'w') as f:
            f.write(json.dumps(roi_struct))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wayside_groups', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gen_default_roi(args)
