import pickle
import argparse
import os
import os.path as osp
import numpy as np
import itertools
import sys
import json
import shutil
from joblib import Parallel, delayed, parallel_backend
from pyquaternion import Quaternion
from typing import List
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.kitti import KittiDB
from typing import cast
from collections import defaultdict

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


def is_in_roi(box_lidar_nusc: Box):
    box_lidar_nusc = box_lidar_nusc.copy()
    nu_lidar_to_kitti = Quaternion(axis=(0, 0, 1), angle=-np.pi / 2)
    box_lidar_nusc.rotate(nu_lidar_to_kitti)
    corners = box_lidar_nusc.corners().T
    min_x, max_x, min_y, max_y = -30., 40.4, -40., 40.
    in_roi = any([c[0] > min_x and c[0] < max_x and c[1] >
                 min_y and c[1] < max_y for c in corners])
    return in_roi


def box_to_string(name, box, bbox_2d, truncation, occlusion, alpha, instance_token):
    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
    yaw = -np.arctan2(v[2], v[0])

    # Prepare output.
    name += ' '
    trunc = '{:.2f} '.format(truncation)
    occ = '{:d} '.format(occlusion)
    a = '{:.2f} '.format(alpha)
    bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(
        bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
    # height, width, length.
    hwl = '{:.2} {:.2f} {:.2f} '.format(box.wlh[2], box.wlh[0], box.wlh[1])
    # x, y, z.
    xyz = '{:.2f} {:.2f} {:.2f} '.format(
        box.center[0], box.center[1], box.center[2])
    y = '{:.2f}'.format(yaw)  # Yaw angle.

    output = name + trunc + occ + a + bb + hwl + xyz + y

    return output


def project_to_2d(box, p_left, height, width):
    box = box.copy()

    # KITTI defines the box center as the bottom center of the object.
    # We use the true center, so we need to adjust half height in negative y direction.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Check that some corners are inside the image.
    corners = np.array(
        [corner for corner in box.corners().T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]),
            np.max(imcorners[0]), np.max(imcorners[1]))

    inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (
        0 <= bbox[0] < width and 0 < bbox[2] <= width)
    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (
        0 <= bbox[0] < width or 0 < bbox[2] <= width)
    if not valid:
        return None

    truncated = valid and not inside
    if truncated:
        _bbox = [0] * 4
        _bbox[0] = max(0, bbox[0])
        _bbox[1] = max(0, bbox[1])
        _bbox[2] = min(width, bbox[2])
        _bbox[3] = min(height, bbox[3])

        truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])
                           ) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        bbox = _bbox
    else:
        truncated = 0.0
    return {"bbox": bbox, "truncated": truncated}


def postprocessing(objs, height, width):
    _map = np.ones((height, width), dtype=np.int8) * -1
    objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
    for i, obj in enumerate(objs):
        _map[int(round(obj["bbox_2d"][1])):int(round(obj["bbox_2d"][3])), int(
            round(obj["bbox_2d"][0])):int(round(obj["bbox_2d"][2]))] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj["bbox_2d"][3] - obj["bbox_2d"][1]) / (
            obj["bbox_2d"][2] - obj["bbox_2d"][0])
        obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))
    return objs


def find_closest_integer_in_ref_arr(query_int: int, ref_arr: np.ndarray):
    closest_ind = np.argmin(np.absolute(ref_arr - query_int))
    closest_int = cast(int, ref_arr[closest_ind])
    int_diff = np.absolute(query_int - closest_int)
    return closest_ind, closest_int, int_diff


def box_pose(box: Box) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = box.rotation_matrix
    mat[:3, 3] = box.center
    return np.linalg.inv(mat)


def get_plane_pose(plane) -> np.ndarray:
    translation = np.array([0., 0., -plane['d']/plane['c']])
    orig_normal = np.array([0., 0., 1.])
    plane_normal = np.array([plane['a'], plane['b'], plane['c']])
    if plane_normal[2] < 0.:
        plane_normal *= -1.
    cross = np.cross(orig_normal, plane_normal)
    rot_axis = cross / np.linalg.norm(cross)
    plane_normal_u = plane_normal / np.linalg.norm(plane_normal)
    rot_angle = np.arccos(
        np.clip(np.dot(orig_normal, plane_normal_u), -1.0, 1.0))
    rot = R.from_rotvec(rot_angle * rot_axis).as_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return mat


def get_region_pose(region) -> np.ndarray:
    translation = np.array(
        [region['center_x'], region['center_y'], region['center_z']])
    rot = R.from_euler('xyz', [0., 0., region['rot_z']]).as_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return mat


def create_foreground_map_from_params(pcl, plane, roi_regions):
    pts = pcl.points
    plane_pose = np.linalg.inv(get_plane_pose(plane))
    road_regions = roi_regions['road_regions']
    road_poses = np.array([np.linalg.inv(get_region_pose(r))
                          for r in road_regions])
    background_regions = roi_regions['background_regions']
    background_poses = np.array(
        [np.linalg.inv(get_region_pose(r))for r in background_regions])

    # Generate flags for the plane
    shifted_pts = plane_pose.dot(
        np.vstack((pts[:3, :], np.ones(pts.shape[1]))))[:3, :]
    above_ground = shifted_pts[2, :] >= 0

    # Generate flags for road regions
    shifted_pts = [pose.dot(np.vstack((pts[:3, :], np.ones(pts.shape[1]))))[
        :3, :] for pose in road_poses]
    in_box_flags = np.array([(shifted_pts[idx][0, :] > -r['size_x'] / 2.) *
                             (shifted_pts[idx][0, :] < r['size_x'] / 2.) *
                             (shifted_pts[idx][1, :] > -r['size_y'] / 2.) *
                             (shifted_pts[idx][1, :] < r['size_y'] / 2.) *
                             (shifted_pts[idx][2, :] > -r['size_z'] / 2.) *
                             (shifted_pts[idx][2, :] < r['size_z'] / 2.)
                             for idx, r in enumerate(road_regions)])
    in_road = np.any(in_box_flags, axis=0)

    # Generate flags for background regions
    shifted_pts = [pose.dot(np.vstack((pts[:3, :], np.ones(pts.shape[1]))))[
        :3, :] for pose in background_poses]
    in_box_flags = np.array([(shifted_pts[idx][0, :] > -r['size_x'] / 2.) *
                             (shifted_pts[idx][0, :] < r['size_x'] / 2.) *
                             (shifted_pts[idx][1, :] > -r['size_y'] / 2.) *
                             (shifted_pts[idx][1, :] < r['size_y'] / 2.) *
                             (shifted_pts[idx][2, :] > -r['size_z'] / 2.) *
                             (shifted_pts[idx][2, :] < r['size_z'] / 2.)
                             for idx, r in enumerate(background_regions)])
    in_background = np.any(in_box_flags, axis=0)

    return (above_ground * in_road * ~in_background).astype(int)


def create_foreground_map_from_gt_boxes(pcl, boxes: List[Box]):
    pts = pcl.points
    n_pts = pts.shape[1]
    box_poses = np.array([box_pose(box) for box in boxes])
    box_sizes = np.array([b.wlh for b in boxes])

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
    head_str = f'# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z rgb timestamp_ns device_id active\nSIZE 4 4 4 4 8 1 1\nTYPE F F F F F U U\nCOUNT 1 1 1 1 1 1 1\nWIDTH {n_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n_points}\nDATA ascii\n'
    points_str = '\n'.join(
        [f'{pcl.points[0][i]:.4} {pcl.points[1][i]:.4} {pcl.points[2][i]:.4} {pcl.points[3][i]} {timestamp*1000} 1 {foreground_map[i]}' for i in range(n_points)])
    return head_str+points_str


class WaysideConverter:
    def __init__(self,
                 output_dir: str,
                 nusc_dir: str,
                 plane_dir: str,
                 roi_dir: str,
                 n_sweeps,
                 lidar_name: str = 'LIDAR_TOP',
                 nusc_version: str = 'v1.0-trainval',
                 parallel_n_jobs: int = 8,
                 split: str = 'train',
                 filter_by_fov: bool = False,
                 ):

        self.output_dir = osp.expanduser(output_dir)
        if os.path.isdir(self.output_dir):
            print(f"output folder {self.output_dir} already exists !!")
            sys.exit()
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version
        self.split = split
        self.parallel_n_jobs = parallel_n_jobs
        self.filter_by_fov = filter_by_fov
        self.n_sweeps = n_sweeps
        self.pcd_folder = osp.join(self.output_dir, 'pointcloud')
        self.plane_dir = osp.join(self.output_dir, 'planes')
        self.roi_dir = osp.join(self.output_dir, 'roi_params')
        shutil.copytree(osp.expanduser(plane_dir), self.plane_dir)
        shutil.copytree(osp.expanduser(roi_dir), self.roi_dir)
        self.kitti_dir = osp.join(self.output_dir, 'kitti_format')
        self.kitti_calib_dir = osp.join(self.kitti_dir, 'calib')
        self.kitti_label_dir = osp.join(self.kitti_dir, 'label_2')
        self.kitti_velo_dir = osp.join(self.kitti_dir, 'velodyne')

        for folder in [self.pcd_folder, self.kitti_calib_dir, self.kitti_label_dir, self.kitti_velo_dir]:
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
        wayside_groups, wayside_cam_groups = self.scene_to_wayside_groups(
            boston_scenes)
        # Filter by the number of frames
        wayside_groups = [x for x in wayside_groups if len(x) >= 200]
        wayside_cam_groups = [x for x in wayside_cam_groups if len(x) >= 200]

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

        for wayside_idx, wayside_tokens in tqdm(list(enumerate(wayside_groups))):
            with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
                Parallel()(delayed(self.process_token_to_kitti)(idx, token, cam_token)
                           for idx, token, cam_token in zip(sample_index_groups[wayside_idx], wayside_tokens, wayside_cam_groups[wayside_idx]))
            with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
                Parallel()(delayed(self.save_nusc_lidar_to_wayside_pcd_file)(idx, wayside_idx, token)
                           for idx, token in zip(sample_index_groups[wayside_idx], wayside_tokens))

    def process_token_to_kitti(self, idx, lidar_token, cam_token):
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        # Get sample data.
        sample_name = "%06d" % idx

        # Retrieve sensor records.
        sd_record_cam = self.nusc.get('sample_data', cam_token)
        sd_record_lid = self.nusc.get('sample_data', lidar_token)
        cs_record_cam = self.nusc.get(
            'calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = self.nusc.get(
            'calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        ego_record_lid = self.nusc.get(
            "ego_pose", sd_record_lid["ego_pose_token"])

        # Combine transformations and convert to KITTI format.
        # Note: cam uses same conventions in KITTI and nuScenes.
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                      inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                      inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        velo_to_cam_kitti = np.dot(
            velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

        # Currently not used.
        # imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
        imu_to_velo_kitti = transform_matrix(
            cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=True
        )
        imu_to_velo_kitti = np.dot(
            kitti_to_nu_lidar_inv.transformation_matrix, imu_to_velo_kitti)
        expected_kitti_imu_to_velo_rot = np.eye(3)
        assert (imu_to_velo_kitti[:3, :3].round(
            0) == expected_kitti_imu_to_velo_rot).all(), imu_to_velo_kitti[:3, :3].round(0)
        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Projection matrix.
        p_left_kitti = np.zeros((3, 4))
        # Cameras are always rectified.
        p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']

        # Create KITTI style transforms.
        velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
        velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

        # Check that the rotation has the same format as in KITTI.
        assert (velo_to_cam_rot.round(0) == np.array(
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
        assert (velo_to_cam_trans[1:3] < 0).all()

        # Retrieve the token from the lidar.
        # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
        # not the camera.
        filename_cam_full = sd_record_cam['filename']
        filename_lid_full = sd_record_lid['filename']

        # Create calibration file.
        kitti_transforms = dict()
        kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
        kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
        # Cameras are already rectified.
        kitti_transforms['R0_rect'] = r0_rect.rotation_matrix
        kitti_transforms['Tr_velo_to_cam'] = np.hstack(
            (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
        kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
        calib_path = os.path.join(self.kitti_calib_dir, sample_name + '.txt')
        with open(calib_path, "w") as calib_file:
            for (key, val) in kitti_transforms.items():
                val = val.flatten()
                val_str = '%.12e' % val[0]
                for v in val[1:]:
                    val_str += ' %.12e' % v
                calib_file.write('%s: %s\n' % (key, val_str))

        # Convert lidar.
        # Note that we are only using a single sweep, instead of the commonly used n sweeps.
        src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
        dst_lid_path = os.path.join(self.kitti_velo_dir, sample_name + '.bin')
        assert not dst_lid_path.endswith('.pcd.bin')
        if self.n_sweeps is not None:
            temp = defaultdict(dict)
            temp['data'][self.lidar_name] = lidar_token
            pcl, _ = LidarPointCloud.from_file_multisweep(
                self.nusc, temp, self.lidar_name, self.lidar_name, nsweeps=self.n_sweeps)
        else:
            pcl = LidarPointCloud.from_file(src_lid_path)
        # pcl, _ = LidarPointCloud.from_file_multisweep_future(self.nusc, sample, self.lidar_name, self.lidar_name, nsweeps=5)
        # In KITTI lidar frame.
        pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)
        with open(dst_lid_path, "w") as lid_file:
            pcl.points.T.tofile(lid_file)
            # recover = np.fromfile(
            #     dst_lid_path, dtype=np.float32).reshape(-1, 4)

        # # Write label file.
        # sample_annotation_tokens = sample['anns']
        label_path = os.path.join(self.kitti_label_dir, sample_name + '.txt')
        if os.path.exists(label_path):
            print('Skipping existing file: %s' % label_path)
            return

        if sd_record_lid['is_key_frame']:
            objects = []
            sample = self.nusc.get("sample", sd_record_lid['sample_token'])
            sample_annotation_tokens = sample['anns']
            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = self.nusc.get(
                    'sample_annotation', sample_annotation_token)

                # Get box in LIDAR frame.
                _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                 selected_anntokens=[sample_annotation_token])
                box_lidar_nusc = box_lidar_nusc[0]

                # Filter by Roi
                if not is_in_roi(box_lidar_nusc):
                    continue

                obj = dict()

                # Convert nuScenes category to nuScenes detection challenge category.
                obj["detection_name"] = category_to_detection_name(
                    sample_annotation['category_name'])

                # Skip categories that are not part of the nuScenes detection challenge.
                if obj["detection_name"] is None or obj["detection_name"] not in CLASS_MAP.keys():
                    continue

                obj["detection_name"] = CLASS_MAP[obj["detection_name"]]

                # Convert from nuScenes to KITTI box format.
                obj["box_cam_kitti"] = KittiDB.box_nuscenes_to_kitti(
                    box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                bbox_2d = project_to_2d(
                    obj["box_cam_kitti"], p_left_kitti, imsize[1], imsize[0])
                if bbox_2d is None:
                    if self.filter_by_fov:
                        continue
                    else:
                        bbox_2d = {'bbox': (-1, -1, -1, -1), 'truncated': -1.}

                obj["bbox_2d"] = bbox_2d["bbox"]
                obj["truncated"] = bbox_2d["truncated"]

                # Set dummy score so we can use this file as result.
                obj["box_cam_kitti"].score = 0

                v = np.dot(obj["box_cam_kitti"].rotation_matrix,
                           np.array([1, 0, 0]))
                rot_y = -np.arctan2(v[2], v[0])
                obj["alpha"] = -np.arctan2(obj["box_cam_kitti"].center[0],
                                           obj["box_cam_kitti"].center[2]) + rot_y
                obj["depth"] = np.linalg.norm(
                    np.array(obj["box_cam_kitti"].center[:3]))
                obj["instance_token"] = sample_annotation['instance_token']
                objects.append(obj)
            if self.filter_by_fov:
                objects = postprocessing(objects, imsize[1], imsize[0])
            else:
                for obj in objects:
                    obj["occluded"] = -1

            with open(label_path, "w") as label_file:
                for obj in objects:
                    # Convert box to output string format.
                    output = box_to_string(name=obj["detection_name"],
                                           box=obj["box_cam_kitti"],
                                           bbox_2d=obj["bbox_2d"],
                                           truncation=obj["truncated"],
                                           occlusion=obj["occluded"],
                                           alpha=obj["alpha"],
                                           instance_token=obj['instance_token'])
                    label_file.write(output + '\n')

    def save_nusc_lidar_to_wayside_pcd_file(self, idx, wayside_idx, token):
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        sample_name = "%06d" % idx
        wayside_name = "%06d" % wayside_idx
        sd_record_lid = self.nusc.get('sample_data', token)
        filename_lid_full = sd_record_lid['filename']
        src_lid_path = osp.join(self.nusc.dataroot, filename_lid_full)
        dst_pcd_path = osp.join(
            self.pcd_folder, sample_name + '.pcd')
        if self.n_sweeps is not None:
            temp = defaultdict(dict)
            temp['data'][self.lidar_name] = token
            pcl, _ = LidarPointCloud.from_file_multisweep(
                self.nusc, temp, self.lidar_name, self.lidar_name, nsweeps=self.n_sweeps)
        else:
            pcl = LidarPointCloud.from_file(src_lid_path)
        # pcl, _ = LidarPointCloud.from_file_multisweep_future(self.nusc, sample, self.lidar_name, self.lidar_name, nsweeps=5)
        # In KITTI lidar frame.
        pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)

        # # Obtaining annotation boxes
        # closest_sample = self.nusc.get(
        #     'sample', sd_record_lid['sample_token'])
        # anno_tokens = closest_sample['anns']

        # _, nusc_lidar_boxes, _ = self.nusc.get_sample_data(
        #     token, box_vis_level=BoxVisibility.NONE)
        # for x in nusc_lidar_boxes:
        #     x.rotate(kitti_to_nu_lidar_inv)

        # # Filter boxes
        # anno_class_map = {x: category_to_detection_name(self.nusc.get(
        #     'sample_annotation', x)['category_name']) for x in anno_tokens}
        # anno_tokens = [x for x in anno_tokens if anno_class_map[x]
        #                is not None and anno_class_map[x] in CLASS_MAP.keys()]
        # nusc_lidar_boxes = [
        #     x for x in nusc_lidar_boxes if x.token in anno_tokens]

        # # Create foreground map based on the boxes
        # foreground_map = create_foreground_map_from_gt_boxes(
        #     pcl, nusc_lidar_boxes)

        plane = json.load(
            open(osp.join(self.plane_dir, wayside_name+'.json'), 'r'))
        roi_regions = json.load(
            open(osp.join(self.roi_dir, wayside_name+'.json'), 'r'))

        foreground_map = create_foreground_map_from_params(
            pcl, plane, roi_regions)

        pcd_content = form_pcd_file_content(
            pcl, foreground_map, sd_record_lid['timestamp'])
        with open(dst_pcd_path, "w") as pcd_file:
            pcd_file.write(pcd_content)

    def scene_to_wayside_groups(self, scenes) -> List[str]:
        wayside_groups = []
        wayside_cam_groups = []
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

            cam_tokens = []
            cam_timestamp = []
            cam_token = first_sample['data']['CAM_FRONT']
            while (cam_token != ''):
                cam_tokens.append(cam_token)
                sample_data = self.nusc.get("sample_data", cam_token)
                cam_timestamp.append(sample_data['timestamp'])
                cam_token = sample_data['next']
            cam_timestamp = np.array(cam_timestamp)
            closest_cam_tokens = []
            for i in range(ld_tokens.shape[0]):
                closest_ind, closest_int, int_diff = find_closest_integer_in_ref_arr(
                    ld_timestamps[i], cam_timestamp)
                closest_cam_tokens.append(cam_tokens[closest_ind])

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
            wayside_cam_groups += [[closest_cam_tokens[y]
                                    for y in x] for x in id_groups]

        return wayside_groups, wayside_cam_groups

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
        '--plane_dir', type=str)
    parser.add_argument(
        '--roi_dir', type=str)
    parser.add_argument(
        '--output_dir', type=str)
    parser.add_argument('--n_sweeps', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wc = WaysideConverter(output_dir=args.output_dir,
                          nusc_dir=args.nusc_dir,
                          n_sweeps=args.n_sweeps,
                          plane_dir=args.plane_dir,
                          roi_dir=args.roi_dir,
                          )
    wc.save_to_wayside_pcds()
