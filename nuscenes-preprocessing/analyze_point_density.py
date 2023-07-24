from kitti import KittiDB
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from os import path as osp
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import argparse
import os
import json


kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)


def get_average_box_points(box_infos, classes, range_start, range_end):
    filtered_box_infos = []
    for box_info in box_infos:
        box = box_info[0]
        distance = (box.center[0] ** 2 + box.center[1] ** 2) ** 0.5
        in_range = distance >= range_start and distance < range_end
        in_class = box.name in classes
        if in_range and in_class:
            filtered_box_infos.append(box_info)
    num_points_list = [x[1] for x in filtered_box_infos]
    # num_points_list = [x[1] / (x[0].wlh[0]*x[0].wlh[1]*x[0].wlh[2])
    #                    for x in filtered_box_infos]
    if len(num_points_list) == 0:
        return None
    percent = len(filtered_box_infos) / len(box_infos) * 100.
    return (sum(num_points_list) / len(num_points_list), percent)


def get_average_size(box_infos, classes, range_start, range_end):
    filtered_box_infos = []
    for box_info in box_infos:
        box = box_info[0]
        distance = (box.center[0] ** 2 + box.center[1] ** 2) ** 0.5
        in_range = distance >= range_start and distance < range_end
        in_class = box.name in classes
        if in_range and in_class:
            filtered_box_infos.append(box_info)
    if len(filtered_box_infos) == 0:
        return None
    size_l = [x[0].wlh[1] for x in filtered_box_infos]
    avg_size_l = sum(size_l) / len(size_l)
    size_w = [x[0].wlh[0] for x in filtered_box_infos]
    avg_size_w = sum(size_w) / len(size_w)
    size_h = [x[0].wlh[2] for x in filtered_box_infos]
    avg_size_h = sum(size_h) / len(size_h)
    return (avg_size_l, avg_size_w, avg_size_h)


def analyze_box_info(box_infos):
    # unique_classes = sorted(list(set([x[0].name for x in box_infos])))
    # print(f'The unique classes are: {unique_classes}')
    # for class_name in unique_classes:
    #     avg_size = get_average_size(box_infos, [class_name], 0, 40)
    #     if avg_size == None:
    #         continue
    #     print(
    #         f'{class_name}: {avg_size[0]:.2} {avg_size[1]:.2} {avg_size[2]:.2}')
    box_infos = [box for box in box_infos if (
        box[0].center[0] ** 2 + box[0].center[1] ** 2) ** 0.5 < 40]
    ranges = [[0, 20], [20, 40], [0, 40]]
    small_classes = ['bicycle', 'motorcycle', 'pedestrian', 'Cyclist']
    medium_classes = ['car', 'Car', 'Truck']
    large_classes = ['bus', 'construction_vehicle', 'trailer', 'truck', 'Bus']
    small_results_str = ''
    medium_results_str = ''
    large_results_str = ''
    for range in ranges:
        small_range_result = get_average_box_points(
            box_infos, small_classes, range[0], range[1])
        if small_range_result is not None:
            small_results_str += f'{small_range_result[0]:.2f}'.rjust(
                12) + f'({small_range_result[1]:.2f}%)'.rjust(8)
        else:
            small_results_str += ''.rjust(20)

        medium_range_result = get_average_box_points(
            box_infos, medium_classes, range[0], range[1])
        if medium_range_result is not None:
            medium_results_str += f'{medium_range_result[0]:.2f}'.rjust(
                12) + f'({medium_range_result[1]:.2f}%)'.rjust(8)
        else:
            medium_results_str += ''.rjust(20)

        large_range_result = get_average_box_points(
            box_infos, large_classes, range[0], range[1])
        if large_range_result is not None:
            large_results_str += f'{large_range_result[0]:.2f}'.rjust(
                12) + f'({large_range_result[1]:.2f}%)'.rjust(8)
        else:
            large_results_str += ''.rjust(20)

    print(''.rjust(20) + '0-20'.rjust(20) +
          '20-40'.rjust(20) + '0-40'.rjust(20))
    print('Small'.rjust(20) + small_results_str)
    print('Medium'.rjust(20) + medium_results_str)
    print('Large'.rjust(20) + large_results_str)


def box_pose(box: Box) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = box.rotation_matrix
    mat[:3, 3] = box.center
    return mat


def analyze_point_density(args):
    kitti_dir = args.kitti_dir
    index_file = args.index_file
    box_infos = []
    result_dict = defaultdict(list)

    kitti = KittiDB(root=kitti_dir)
    tokens = kitti.tokens

    if index_file is not None:
        indices = open(index_file, 'r').read().splitlines()
        tokens = [x for x in tokens if x.split('_')[-1] in indices]

    for token in tqdm(tokens):
        index = token.split('_')[-1]
        boxes, _truncs = kitti.get_boxes(token)
        bin_path = kitti.get_filepath(token, 'velodyne', root=kitti_dir)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4).T
        for box in boxes:
            box.rotate(kitti_to_nu_lidar.inverse)
            pose_inv = np.linalg.inv(box_pose(box))
            shifted_pts = pose_inv.dot(
                np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]
            in_box_flag = (shifted_pts[0, :] > -box.wlh[1] / 2.) * \
                (shifted_pts[0, :] < box.wlh[1] / 2.) * \
                (shifted_pts[1, :] > -box.wlh[0] / 2.) * \
                (shifted_pts[1, :] < box.wlh[0] / 2.) * \
                (shifted_pts[2, :] > -box.wlh[2] / 2.) * \
                (shifted_pts[2, :] < box.wlh[2] / 2.)
            num_box_points = sum(in_box_flag)
            box_infos.append((box, num_box_points))
        analyze_box_info(box_infos)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, default=None)
    parser.add_argument('--index_file', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_point_density(args)
