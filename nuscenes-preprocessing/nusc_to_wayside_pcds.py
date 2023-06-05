import pickle
import argparse
import os
import os.path as osp
import numpy as np
import itertools
import sys
import json
from pyquaternion import Quaternion
from typing import List
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name

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


def box_pose(box: Box) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = box.rotation_matrix
    mat[:3, 3] = box.center
    return np.linalg.inv(mat)


def create_foreground_map_from_gt_boxes(pcl, boxes: List[Box]) -> int:
    pts = pcl.points
    n_pts = pts.shape[1]
    box_poses = np.array([box_pose(box) for box in boxes])
    box_sizes = np.array([b.wlh for b in boxes])
    # (N_boxes, 4, N_pts)
    # shifted_pts = box_poses @ np.vstack((pts[:3, :], np.ones(pts.shape[1])))
    # in_box_flags = (shifted_pts[:, 0, :] > np.tile(-box_sizes[:, 1] / 2., (n_pts, 1)).T) * \
    #     (shifted_pts[:, 0, :] < np.tile(box_sizes[:, 1] / 2., (n_pts, 1)).T) * \
    #     (shifted_pts[:, 1, :] > np.tile(-box_sizes[:, 0] / 2., (n_pts, 1)).T) * \
    #     (shifted_pts[:, 1, :] < np.tile(box_sizes[:, 0] / 2., (n_pts, 1)).T) * \
    #     (shifted_pts[:, 2, :] > np.tile(-box_sizes[:, 2] / 2., (n_pts, 1)).T) * \
    #     (shifted_pts[:, 2, :] < np.tile(box_sizes[:, 2] / 2., (n_pts, 1)).T)

    shifted_pts = [pose.dot(np.vstack((pts[:3, :], np.ones(pts.shape[1]))))[
        :3, :] for pose in box_poses]
    in_box_flags = np.array([(shifted_pts[idx][0, :] > -b.wlh[1] / 2.) *
                             (shifted_pts[idx][0, :] < b.wlh[1] / 2.) *
                             (shifted_pts[idx][1, :] > -b.wlh[0] / 2.) *
                             (shifted_pts[idx][1, :] < b.wlh[0] / 2.) *
                             (shifted_pts[idx][2, :] > -b.wlh[2] / 2.) *
                             (shifted_pts[idx][2, :] < b.wlh[2] / 2.)
                             for idx, b in enumerate(boxes)])
    return np.any(in_box_flags, axis=0).astype(int)


def form_trans_mat(translation, rotation):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def form_pcd_file_content(pcl: LidarPointCloud, foreground_map,  timestamp) -> str:
    n_points = pcl.points.shape[1]
    head_str = f'# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb timestamp_ns device_id active\nSIZE 4 4 4 4 8 1 1\nTYPE F F F U F U U\nCOUNT 1 1 1 1 1 1 1\nWIDTH {n_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n_points}\nDATA ascii\n'
    points_str = '\n'.join(
        [f'{pcl.points[0][i]:.4} {pcl.points[1][i]:.4} {pcl.points[2][i]:.4} 0 {timestamp*1000} 1 {foreground_map[i]}' for i in range(n_points)])
    return head_str+points_str


class WaysideConverter:
    def __init__(self,
                 output_dir: str,
                 nusc_dir: str,
                 lidar_name: str = 'LIDAR_TOP',
                 nusc_version: str = 'v1.0-trainval',
                 parallel_n_jobs: int = 8,
                 split: str = 'train'
                 ):

        self.output_dir = osp.expanduser(output_dir)
        if os.path.isdir(self.output_dir):
            print(f"output folder {self.output_dir} already exists !!")
            sys.exit()
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version
        self.split = split
        self.parallel_n_jobs = parallel_n_jobs
        self.pcd_folder = osp.join(self.output_dir, 'pointcloud')

        for folder in [self.pcd_folder]:
            os.makedirs(folder)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_dir)

        print("self.nusc.sample", len(self.nusc.sample))

    def save_to_wayside_pcds(self) -> None:

        # Use only the samples from the current split.
        scene_splits = create_splits_scenes(verbose=False)
        scene_to_log = {scene['name']: (scene,
                                        self.nusc.get('log', scene['log_token'])[
                                            'logfile'],
                                        self.nusc.get('log', scene['log_token'])['location']) for scene in self.nusc.scene}
        scenes = scene_splits[self.split]
        boston_scenes = []
        for scene in scenes:
            if scene_to_log[scene][2].startswith('boston'):
                boston_scenes.append(scene_to_log[scene][0])
        np.random.seed(1024)
        np.random.shuffle(boston_scenes)

        # Pickup wayside data
        wayside_groups = self.scene_to_wayside_groups(boston_scenes)
        # Filter by the number of frames
        wayside_groups = [x for x in wayside_groups if len(x) >= 200]

        sample_index_groups = []
        token_map = {}
        cnt = 0
        for group in wayside_groups:
            sample_index_groups.append([])
            for token in group:
                sample_index_groups[-1].append(cnt)
                token_map[cnt] = token
                cnt += 1

        # Create index map from wayside index to kitti ground-truth index
        scene_tokens = [scene['token'] for scene in boston_scenes]
        anno_sample_tokens = self.split_to_samples_annotated(scene_tokens)
        anno_token_index_map = {token: idx for idx,
                                token in enumerate(anno_sample_tokens)}
        wayside2gt_index_map = {wayside_idx: anno_token_index_map[token_map[wayside_idx]] for wayside_idx in range(
            cnt) if token_map[wayside_idx] in anno_token_index_map.keys()}

        # Save meta data
        with open(osp.join(self.output_dir, 'wayside_groups.json'), 'w') as f:
            f.write(json.dumps(sample_index_groups, indent=4))
        with open(osp.join(self.output_dir, 'token_map.json'), 'w') as f:
            f.write(json.dumps(token_map, indent=4))
        with open(osp.join(self.output_dir, 'wayside2gt_map.json'), 'w') as f:
            f.write(json.dumps(wayside2gt_index_map, indent=4))

        for wayside_idx, wayside_tokens in enumerate(tqdm(wayside_groups)):
            self.save_wayside_points(
                wayside_idx, wayside_tokens, sample_index_groups[wayside_idx])

        # Save lidar points in pcd format

    def save_wayside_points(self, wayside_idx, wayside_tokens, sample_indices):
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        for idx, token in enumerate(tqdm(wayside_tokens)):
            sample_index = sample_indices[idx]
            sample_name = "%06d" % sample_index
            sd_record_lid = self.nusc.get('sample_data', token)
            filename_lid_full = sd_record_lid['filename']
            src_lid_path = osp.join(self.nusc.dataroot, filename_lid_full)
            dst_pcd_path = osp.join(
                self.pcd_folder, sample_name + '.pcd')
            pcl = LidarPointCloud.from_file(src_lid_path)
            # pcl, _ = LidarPointCloud.from_file_multisweep_future(self.nusc, sample, self.lidar_name, self.lidar_name, nsweeps=5)
            # In KITTI lidar frame.
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)

            # Obtaining annotation boxes
            closest_sample = self.nusc.get(
                'sample', sd_record_lid['sample_token'])
            anno_tokens = closest_sample['anns']

            _, nusc_lidar_boxes, _ = self.nusc.get_sample_data(
                token, box_vis_level=BoxVisibility.NONE)
            for x in nusc_lidar_boxes:
                x.rotate(kitti_to_nu_lidar_inv)

            # Filter boxes
            anno_class_map = {x: category_to_detection_name(self.nusc.get(
                'sample_annotation', x)['category_name']) for x in anno_tokens}
            anno_tokens = [x for x in anno_tokens if anno_class_map[x]
                           is not None and anno_class_map[x] in CLASS_MAP.keys()]
            nusc_lidar_boxes = [
                x for x in nusc_lidar_boxes if x.token in anno_tokens]

            # Create foreground map based on the boxes
            foreground_map = create_foreground_map_from_gt_boxes(
                pcl, nusc_lidar_boxes)

            pcd_content = form_pcd_file_content(
                pcl, foreground_map, sd_record_lid['timestamp'])
            with open(dst_pcd_path, "w") as pcd_file:
                pcd_file.write(pcd_content)

    def scene_to_wayside_groups(self, scenes) -> List[str]:
        wayside_groups = []
        speed_thresh = 0.5
        for scene in scenes:
            first_sample_token = scene['first_sample_token']
            first_sample = self.nusc.get("sample", first_sample_token)
            ld_token = first_sample['data'][self.lidar_name]
            ld_tokens = []
            ld_timestamps = []
            ld_poses = []
            while (ld_token != ''):
                ld_tokens.append(ld_token)
                sample_data = self.nusc.get("sample_data", ld_token)
                ld_timestamps.append(sample_data['timestamp'])
                ego_record_lid = self.nusc.get(
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

                is_stopped = avg_speed <= speed_thresh
                stopped_flags.append(is_stopped)
            id_groups = [list(g) for k, g in itertools.groupby(
                range(len(stopped_flags)), lambda x: stopped_flags[x]) if k]
            wayside_groups += [[ld_tokens[y]
                                for y in x] for x in id_groups]

        return wayside_groups

    def split_to_samples_annotated(self, split_tokens: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        annotated_ld_tokens = []
        for sample in self.nusc.sample:
            if sample['scene_token'] not in split_tokens:
                continue
            lidar_token = sample['data'][self.lidar_name]
            annotated_ld_tokens.append(lidar_token)
        return annotated_ld_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nusc_dir', type=str)
    parser.add_argument(
        '--output_dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wc = WaysideConverter(output_dir=args.output_dir,
                          nusc_dir=args.nusc_dir,
                          )
    wc.save_to_wayside_pcds()
