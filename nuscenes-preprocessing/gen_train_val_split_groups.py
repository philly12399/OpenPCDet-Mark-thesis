import pickle
import argparse
import os.path as osp
import os
import sys
import json
import numpy as np
import itertools
import random
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modest_output_dir", type=str)
    parser.add_argument("--wayside_output_dir", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    modest_output_dir = args.modest_output_dir
    modest_token_map = osp.join(
        modest_output_dir, '../../meta_data/nuscenes/labeled_tokens.pkl')
    wayside_output_dir = args.wayside_output_dir
    wayside_token_map = osp.join(wayside_output_dir, 'token_map.json')
    wayside_groups = osp.join(wayside_output_dir, 'wayside_groups.json')
    # Create index map
    index_map = {}
    modest_train_indices = [int(x.split('.')[0])
                            for x in os.listdir(osp.join(modest_output_dir))]
    modest_token_map = pickle.load(open(modest_token_map, 'rb'))
    wayside_token_map = json.load(open(wayside_token_map, 'r'))
    wayside_groups = json.load(open(wayside_groups, 'r'))
    wayside_groups = [[int(y) for y in x] for x in wayside_groups]
    wayside_token_map = {v: k for k, v in wayside_token_map.items()}
    index_map = {idx: wayside_token_map[token] for idx, token in enumerate(
        modest_token_map) if token in wayside_token_map and idx in modest_train_indices}

    modest_wayside_indices = list(map(int, index_map.values()))
    group_counts = defaultdict(int)
    for idx in modest_wayside_indices:
        wayside_id = [way_id for way_id, frames in enumerate(
            wayside_groups) if idx in frames][0]
        group_counts[wayside_id] += 1

    random.seed(10)
    val_wayside_number = 20
    wayside_gt_frames = [int(x.split('.')[0])
                         for x in os.listdir(osp.join(wayside_output_dir, 'kitti_format/label_2'))]
    val_wayside_ids = random.sample(
        [x for x, _ in enumerate(wayside_groups) if x not in group_counts], k=val_wayside_number)
    val_wayside_ids.sort()
    val_frames = [y for x in val_wayside_ids for y in wayside_groups[x]]
    val_gt_frames = [x for x in wayside_gt_frames if x in val_frames]
    val_gt_frames.sort()
    train_gt_frames = modest_wayside_indices
    train_gt_frames.sort()

    train_wayside_ids = list(group_counts.keys())
    train_wayside_ids.sort()
    print(train_wayside_ids, len(train_gt_frames))
    print(val_wayside_ids, len(val_gt_frames))

    with open(osp.join('params', 'train.txt'), 'w') as f:
        output = '\n'.join([f'{x:06}' for x in train_gt_frames])
        f.write(output)

    with open(osp.join('params', 'val.txt'), 'w') as f:
        output = '\n'.join([f'{x:06}' for x in val_gt_frames])
        f.write(output)

    # for k, v in group_counts.items():
    #     print(k, v)
    # print(len(group_counts))
    # output_dir = osp.join(osp.dirname(modest_output_dir),
    #                       osp.basename(modest_output_dir)+'_wayside_indexed')
    # if os.path.isdir(output_dir):
    #     print(f"output folder {output_dir} already exists !!")
    #     sys.exit()

    # output_folders = ['label_2', 'calib', 'velodyne', 'image_2']
    # suffix_map = {
    #     'label_2': 'txt',
    #     'calib': 'txt',
    #     'velodyne': 'bin',
    #     'image_2': 'png',
    # }

    # for folder in output_folders:
    #     os.makedirs(osp.join(output_dir, folder))

    # for folder in output_folders:
    #     for src_idx, dst_idx in index_map.items():
    #         src_name = f'{int(src_idx):06}.{suffix_map[folder]}'
    #         dst_name = f'{int(dst_idx):06}.{suffix_map[folder]}'
    #         src_path = osp.join(osp.join(modest_output_dir, folder), src_name)
    #         dst_path = osp.join(osp.join(output_dir, folder), dst_name)
    #         print(f'linking from {src_path} to {dst_path}.')
    #         os.symlink(src_path, dst_path)
