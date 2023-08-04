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
    parser.add_argument("--modest_output_dir", type=str)
    parser.add_argument("--modest_token_map", type=str)
    parser.add_argument("--wayside_token_map", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    modest_output_dir = args.modest_output_dir
    modest_token_map = args.modest_token_map
    wayside_token_map = args.wayside_token_map
    # Create index map
    index_map = {}
    modest_train_indices = [int(x.split('.')[0]) for x in os.listdir(
        osp.join(modest_output_dir, 'label_2'))]
    modest_token_map = pickle.load(open(modest_token_map, 'rb'))
    wayside_token_map = json.load(open(wayside_token_map, 'r'))
    wayside_token_map = {v: k for k, v in wayside_token_map.items()}
    index_map = {idx: wayside_token_map[token] for idx, token in enumerate(
        modest_token_map) if token in wayside_token_map and idx in modest_train_indices}

    output_dir = osp.join(osp.dirname(modest_output_dir),
                          osp.basename(modest_output_dir)+'_wayside_indexed')
    if os.path.isdir(output_dir):
        print(f"output folder {output_dir} already exists !!")
        sys.exit()

    output_folders = ['label_2', 'calib']
    suffix_map = {
        'label_2': 'txt',
        'calib': 'txt',
        'velodyne': 'bin',
        'image_2': 'png',
    }

    for folder in output_folders:
        os.makedirs(osp.join(output_dir, folder))

    for folder in output_folders:
        for src_idx, dst_idx in index_map.items():
            src_name = f'{int(src_idx):06}.{suffix_map[folder]}'
            dst_name = f'{int(dst_idx):06}.{suffix_map[folder]}'
            src_path = osp.join('../../', osp.basename(modest_output_dir), folder, src_name)
            dst_path = osp.join(osp.join(output_dir, folder), dst_name)
            print(f'linking from {src_path} to {dst_path}.')
            os.symlink(src_path, dst_path)
