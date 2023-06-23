from kitti import KittiDB
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from os import path as osp
from tqdm import tqdm

import numpy as np
import argparse
import os
import json


kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)


def filter_boxes(boxes, min_ranges, max_ranges):
    output_boxes = []
    for box in boxes:
        corners = box.corners().T
        flags = (corners[:, 0] > min_ranges[0]) * \
                (corners[:, 1] > min_ranges[1]) *\
                (corners[:, 0] < max_ranges[0]) *\
                (corners[:, 1] < max_ranges[1])
        if any(flags):
            output_boxes.append(box)
    return output_boxes


def gen_ground_planes(args):
    gt_kitti_dir = args.gt_kitti_dir
    output_dir = args.output_dir
    wayside_groups = json.load(open(args.wayside_groups, 'r'))
    imsize = (900, 1600)

    if osp.isdir(output_dir):
        print(f"output dir {output_dir} already exists.")
        return

    kitti = KittiDB(root=gt_kitti_dir)

    os.mkdir(output_dir)

    for wayside_id, indices in enumerate(tqdm(wayside_groups)):
        indices = [int(x) for x in indices]
        tokens = [x for x in kitti.tokens if int(x.split('_')[-1]) in indices]
        bottom_vertices = np.zeros((0, 3))
        for token in tokens:
            boxes, _truncs = kitti.get_boxes(token)
            for box in boxes:
                box.rotate(kitti_to_nu_lidar.inverse)

            min_ranges = [-30., -40.]
            max_ranges = [40.4, 40.]
            # Filter boxes by range
            temp = len(boxes)
            boxes = filter_boxes(boxes, min_ranges, max_ranges)
            bottom_vertices = np.append(bottom_vertices, np.vstack(
                [x.corners().T[[2, 3, 6, 7], :] for x in boxes]), axis=0)
        # Denote the plane equation as: z = mx + ny + t
        x = bottom_vertices[:, 0]
        y = bottom_vertices[:, 1]
        A = np.vstack([x, y, np.ones(len(x))]).T
        z = bottom_vertices[:, 2]
        # Solve the least-square solution for Ax = b
        m, n, t = np.linalg.lstsq(A, z, rcond=None)[0]
        # Denote the plane equation as: 0 = ax + by + cz + d
        a, b, c, d = m, n, -1., t
        output_data = {'a': a, 'b': b, 'c': c, 'd': d}
        output_path = osp.join(output_dir, f'{wayside_id:06}.json')
        with open(output_path, 'w') as f:
            f.write(json.dumps(output_data, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_kitti_dir', type=str, default=None)
    parser.add_argument('--wayside_groups', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gen_ground_planes(args)
