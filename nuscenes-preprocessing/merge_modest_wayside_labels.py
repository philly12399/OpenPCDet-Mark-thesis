from kitti import KittiDB
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from os import path as osp
from tqdm import tqdm
from shapely.geometry import Polygon

import numpy as np
import argparse
import os
import json


kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)


def box_to_polygon(box: Box):
    corners = box.corners().T
    coords = corners[[0, 4, 5, 1, 0], :2]
    poly = Polygon(coords)
    return poly


def iou3d(box_a: Box, box_b: Box):
    poly_a = box_to_polygon(box_a)
    poly_b = box_to_polygon(box_b)
    intersec = poly_a.intersection(poly_b).area
    return intersec / (poly_a.area + poly_b.area - intersec)


def merge_modest_wayside_labels(args):
    modest_kitti_dir = args.modest_dir
    wayside_kitti_dir = args.wayside_dir
    output_dir = osp.join(osp.abspath(
        osp.join(wayside_kitti_dir, os.pardir)), 'merged_kitti_format')
    if osp.isdir(output_dir):
        print(f"output dir {output_dir} already exists.")
        return
    os.mkdir(output_dir)
    os.mkdir(osp.join(output_dir, 'label_2'))
    os.symlink(osp.join(wayside_kitti_dir, 'calib'),
               osp.join(output_dir, 'calib'))
    os.symlink(osp.join(wayside_kitti_dir, 'velodyne'),
               osp.join(output_dir, 'velodyne'))

    wayside_kitti = KittiDB(root=wayside_kitti_dir)
    modest_kitti = KittiDB(root=modest_kitti_dir)

    common_tokens = list(set(wayside_kitti.tokens).intersection(
        set(modest_kitti.tokens)))
    common_tokens.sort()

    for token in tqdm(common_tokens):
        wayside_boxes, _ = wayside_kitti.get_boxes(token)
        modest_boxes, _ = modest_kitti.get_boxes(token)
        modest_boxes_filtered = [x for x in modest_boxes if sum(
            [iou3d(x, y) for y in wayside_boxes]) == 0]
        merged_boxes = wayside_boxes + modest_boxes_filtered
        for box in merged_boxes:
            box.name = 'Dynamic'
        output_str = boxes_to_string(merged_boxes, wayside_kitti, token)
        output_path = wayside_kitti.get_filepath(
            token, 'label_2', root=output_dir)
        with open(output_path, 'w') as f:
            f.write(output_str)
    print(f"Merge to dir: {output_dir}")


def boxes_to_string(boxes, kitti: KittiDB, token):
    transforms = kitti.get_transforms(token, root=kitti.root)
    vel_to_cam_rot = Quaternion(matrix=transforms['velo_to_cam']['R'])
    velo_to_cam_trans = transforms['velo_to_cam']['T']
    r0_rect = Quaternion(matrix=transforms['r0_rect'])
    boxes = [kitti.box_nuscenes_to_kitti(
        box, vel_to_cam_rot, velo_to_cam_trans, r0_rect) for box in boxes]
    output = "\n".join([kitti.box_to_string(box) for box in boxes])
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modest_dir', type=str, default=None)
    parser.add_argument('--wayside_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge_modest_wayside_labels(args)
