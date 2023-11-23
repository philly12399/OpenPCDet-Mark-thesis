from kitti import KittiDB
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name
from os import path as osp
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

import numpy as np
import argparse
import os
import json
import itertools


class BoxInfo:
    def __init__(self, box, timestamp):
        self.box = box
        self.timestamp = timestamp


kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)

CLASS_MAP = {
    "bicycle": "Dynamic",
    "bus": "Dynamic",
    "car": "Dynamic",
    "construction_vehicle": "Dynamic",
    "motorcycle": "Dynamic",
    "pedestrian": "Dynamic",
    "trailer": "Dynamic",
    "truck": "Dynamic"
}


def box_statistics(object_box_map, fix_time):
    move_distance_map = defaultdict(int)
    for key, box_infos in tqdm(object_box_map.items()):
        if not fix_time:
            max_move_dist = 0
            for i in range(len(box_infos)-1):
                move_dist = np.linalg.norm(
                    box_infos[i].box.center-box_infos[i+1].box.center)
                max_move_dist += move_dist
        else:
            max_move_dist = None
            for i in range(len(box_infos)):
                for j in range(i+1, len(box_infos)):
                    time_diff = (box_infos[j].timestamp -
                                 box_infos[i].timestamp) / 1_000_000
                    if time_diff < 10.:
                        total_distance = sum([np.linalg.norm(
                            box_infos[x].box.center-box_infos[x+1].box.center) for x in range(i, j)])
                        if max_move_dist == None or total_distance > max_move_dist:
                            max_move_dist = total_distance
            if max_move_dist == None:
                continue

        if max_move_dist < 1:
            move_distance_map['~1'] += 1
        elif max_move_dist < 10:
            move_distance_map['1~10'] += 1
        else:
            move_distance_map['10~'] += 1
    move_dist_keys = list(move_distance_map.keys())
    move_dist_keys.sort()
    total_count = sum(list(move_distance_map.values()))
    for dist_key in move_dist_keys:
        print(
            f'{dist_key}: {move_distance_map[dist_key]} ({move_distance_map[dist_key]/total_count*100:.2f}%)')


def is_in_roi(box_lidar_nusc: Box):
    # box_lidar_nusc = box_lidar_nusc.copy()
    # nu_lidar_to_kitti = Quaternion(axis=(0, 0, 1), angle=-np.pi / 2)
    # box_lidar_nusc.rotate(nu_lidar_to_kitti)
    corners = box_lidar_nusc.corners().T
    min_x, max_x, min_y, max_y = -30., 40.4, -40., 40.
    in_roi = any([c[0] > min_x and c[0] < max_x and c[1] >
                 min_y and c[1] < max_y for c in corners])
    return in_roi


def form_trans_mat(translation, rotation):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def scene_to_wayside_groups(nusc, scenes):
    wayside_groups = []
    speed_thresh = 0.5
    for scene in scenes:
        first_sample_token = scene['first_sample_token']
        first_sample = nusc.get("sample", first_sample_token)
        ld_token = first_sample['data']['LIDAR_TOP']
        ld_tokens = []
        ld_timestamps = []
        ld_poses = []
        while (ld_token != ''):
            ld_tokens.append(ld_token)
            sample_data = nusc.get("sample_data", ld_token)
            ld_timestamps.append(sample_data['timestamp'])
            ego_record_lid = nusc.get(
                "ego_pose", sample_data["ego_pose_token"])
            ld_poses.append(form_trans_mat(
                ego_record_lid['translation'], ego_record_lid['rotation']))

            ld_token = sample_data['next']
        ld_tokens = np.array(ld_tokens)

        stopped_flags = []
        for idx, ld_token in enumerate(ld_tokens):
            if idx == len(ld_tokens)-1:
                break
            # The timetamps is recorded in microseconds
            curr_time = ld_timestamps[idx]
            next_time = ld_timestamps[idx+1]
            time_diff_sec = (next_time - curr_time) / 1_000_000
            assert time_diff_sec > 0.02, time_diff_sec

            curr_lidar_pose = ld_poses[idx]
            next_lidar_pose = ld_poses[idx+1]
            pose_diff = next_lidar_pose[:3, 3] - curr_lidar_pose[:3, 3]
            avg_velo = pose_diff / time_diff_sec
            avg_speed = np.linalg.norm(avg_velo)

            curr_lidar_rot = R.from_matrix(
                curr_lidar_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
            next_lidar_rot = R.from_matrix(
                next_lidar_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
            rot_diff = next_lidar_rot - curr_lidar_rot
            avg_rot_speed = rot_diff / time_diff_sec

            is_stopped = avg_speed <= speed_thresh and avg_rot_speed <= 1.
            stopped_flags.append(is_stopped)
        id_groups = [list(g) for k, g in itertools.groupby(
            range(len(stopped_flags)), lambda x: stopped_flags[x]) if k]
        wayside_groups += [[ld_tokens[y]
                            for y in x] for x in id_groups]

    return wayside_groups


def get_box_groups_in_nusc(nusc_dir, index_file, fix_time):
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_dir)
    scene_splits = create_splits_scenes(verbose=False)
    scene_to_log = {scene['name']: (scene,
                                    nusc.get('log', scene['log_token'])[
                                        'logfile'],
                                    nusc.get('log', scene['log_token'])['location']) for scene in nusc.scene}
    scenes = scene_splits['train']
    boston_scenes = []
    for scene in scenes:
        if scene_to_log[scene][2].startswith('boston'):
            boston_scenes.append(scene_to_log[scene][0])
    np.random.seed(1024)
    np.random.shuffle(boston_scenes)

    wayside_groups = scene_to_wayside_groups(nusc, boston_scenes)
    wayside_groups = [x for x in wayside_groups if len(x) >= 200]
    sample_index_groups = []
    cnt = 0
    for group in wayside_groups:
        sample_index_groups.append([])
        for _token in group:
            sample_index_groups[-1].append(cnt)
            cnt += 1

    for i in range(len((wayside_groups))):
        valid_indexes = [j for j in range(len(wayside_groups[i])) if nusc.get(
            'sample_data', wayside_groups[i][j])['is_key_frame']]
        wayside_groups[i] = [wayside_groups[i][j] for j in valid_indexes]
        sample_index_groups[i] = [sample_index_groups[i][j]
                                  for j in valid_indexes]

    if index_file is not None:
        indices = open(index_file, 'r').read().splitlines()
        indices = [int(x) for x in indices]
        for i in range(len((wayside_groups))):
            valid_indexes = [
                j for j, x in enumerate(sample_index_groups[i]) if x in indices]
            wayside_groups[i] = [wayside_groups[i][j] for j in valid_indexes]
            sample_index_groups[i] = [sample_index_groups[i][j]
                                      for j in valid_indexes]
    wayside_groups = [x for x in wayside_groups if len(x) > 0]
    sample_index_groups = [x for x in sample_index_groups if len(x) > 0]

    object_box_map = defaultdict(list)
    print(f"{sum([len(x) for x in wayside_groups])} frames to analyze...")
    all_tokens = [y for x in wayside_groups for y in x]
    for token in tqdm(all_tokens):
        sd_record_lid = nusc.get('sample_data', token)
        sample = nusc.get("sample", sd_record_lid['sample_token'])
        sample_annotation_tokens = sample['anns']
        sample_timestamp = sample['timestamp']

        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nusc.get(
                'sample_annotation', sample_annotation_token)
            instance_token = sample_annotation['instance_token']
            # Get box in LIDAR frame.
            _, box_lidar_nusc, _ = nusc.get_sample_data(token, box_vis_level=BoxVisibility.NONE,
                                                        selected_anntokens=[sample_annotation_token])
            box_lidar_nusc = box_lidar_nusc[0]
            box_lidar_nusc.rotate(kitti_to_nu_lidar.inverse)

            if not is_in_roi(box_lidar_nusc):
                continue

            class_name = category_to_detection_name(
                sample_annotation['category_name'])
            if class_name is None or class_name not in CLASS_MAP.keys():
                continue
            box_info = BoxInfo(box_lidar_nusc, sample_timestamp)
            object_box_map[instance_token].append(box_info)
    box_statistics(object_box_map, fix_time)


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


def get_box_groups_in_super_dir(super_dir, fix_time):
    object_box_map = defaultdict(list)

    files = os.listdir(super_dir)
    files.sort()
    print(f"{len(files)} frames to analyze...")
    for file in tqdm(files):
        index = file.split('.')[0]
        fake_timestamp = int(index) * 100000
        annos_json = json.load(open(osp.join(super_dir, file), 'r'))
        figures = annos_json['figures']
        for fig in figures:
            key = fig['objectKey']
            box = super_figure_to_box(fig)
            box_info = BoxInfo(box, fake_timestamp)
            object_box_map[key].append(box_info)

    box_statistics(object_box_map, fix_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_dir', type=str, default=None)
    parser.add_argument('--super_dir', type=str, default=None)
    parser.add_argument('--index_file', type=str, default=None)
    parser.add_argument('--fix_time', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.nusc_dir is not None:
        get_box_groups_in_nusc(args.nusc_dir, args.index_file, args.fix_time)
    elif args.super_dir is not None:
        get_box_groups_in_super_dir(args.super_dir, args.fix_time)
