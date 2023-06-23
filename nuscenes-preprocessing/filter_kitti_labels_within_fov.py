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

# From boxes in kitti-lidar coordinates to output string


def boxes_to_string(boxes, kitti: KittiDB, token):
    transforms = kitti.get_transforms(token, root=kitti.root)
    vel_to_cam_rot = Quaternion(matrix=transforms['velo_to_cam']['R'])
    velo_to_cam_trans = transforms['velo_to_cam']['T']
    r0_rect = Quaternion(matrix=transforms['r0_rect'])
    boxes = [kitti.box_nuscenes_to_kitti(
        box, vel_to_cam_rot, velo_to_cam_trans, r0_rect) for box in boxes]
    output = "\n".join([kitti.box_to_string(box) for box in boxes])
    return output


def is_in_fov(box: Box, height, width, kitti: KittiDB, token):
    transforms = kitti.get_transforms(token, root=kitti.root)
    velo_to_cam_rot = Quaternion(matrix=transforms['velo_to_cam']['R'])
    velo_to_cam_trans = transforms['velo_to_cam']['T']
    p_left = transforms['p_left']
    r0_rect = Quaternion(matrix=transforms['r0_rect'])

    box = box.copy()
    box = KittiDB.box_nuscenes_to_kitti(
        box, velo_to_cam_rot, velo_to_cam_trans, r0_rect)

    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    corners = np.array(
        [corner for corner in box.corners().T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]),
            np.max(imcorners[0]), np.max(imcorners[1]))

    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (
        0 <= bbox[0] < width or 0 < bbox[2] <= width)
    return valid


def filter_kitti_annos_by_fov(args):
    kitti_dir = args.kitti_dir
    imsize = (900, 1600)

    output_dir = osp.join(osp.dirname(kitti_dir),
                          osp.basename(kitti_dir)+'_filtered_fov')
    if osp.isdir(output_dir):
        print(f"output dir {output_dir} already exists.")
        return

    kitti = KittiDB(root=kitti_dir)

    os.mkdir(output_dir)
    os.symlink(osp.join(kitti_dir, 'calib'), osp.join(output_dir, 'calib'))
    os.symlink(osp.join(kitti_dir, 'velodyne'),
               osp.join(output_dir, 'velodyne'))
    os.mkdir(osp.join(output_dir, 'label_2'))

    for token in tqdm(kitti.tokens):
        index = token.split('_')[-1]
        boxes, _truncs = kitti.get_boxes(token)
        filtered_boxes = [box for box in boxes if is_in_fov(
            box, imsize[0], imsize[1], kitti, token)]

        output_string = boxes_to_string(filtered_boxes, kitti, token)
        output_path = kitti.get_filepath(token, 'label_2', root=output_dir)
        with open(output_path, 'w') as f:
            f.write(output_string)

# from super annos to box in nuscenes lidar coord.


def super_figure_to_box(figure):
    geometry = figure['geometry']
    center = [float(geometry['position']['x']), float(
        geometry['position']['y']), float(geometry['position']['z'])]
    size_wlh = [float(geometry['dimensions']['x']), float(
        geometry['dimensions']['y']), float(geometry['dimensions']['z'])]
    orientation = Quaternion(axis=(0.0, 0.0, 1.0),
                             radians=geometry['rotation']['z'])
    box = Box(center, size_wlh, orientation)
    box.rotate(kitti_to_nu_lidar)
    return box


def filter_super_annos_by_fov(args):
    kitti_dir = args.ref_kitti_dir
    super_dir = args.super_dir
    kitti = KittiDB(root=kitti_dir)
    imsize = (900, 1600)

    files = os.listdir(super_dir)
    files.sort()
    for file in tqdm(files):
        index = file.split('.')[0]
        annos_json = json.load(open(osp.join(super_dir, file), 'r'))
        figures = annos_json['figures']
        keys = [x['objectKey'] for x in figures]
        boxes = [super_figure_to_box(f) for f in figures]
        filtered_keys = [key for key, box in zip(keys, boxes) if is_in_fov(
            box, imsize[0], imsize[1], kitti, f'train_{index}')]
        print(f'{len(keys)} -> {len(filtered_keys)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, default=None)
    parser.add_argument('--ref_kitti_dir', type=str, default=None)
    parser.add_argument('--super_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.super_dir is not None:
        filter_super_annos_by_fov(args)
    else:
        filter_kitti_annos_by_fov(args)
