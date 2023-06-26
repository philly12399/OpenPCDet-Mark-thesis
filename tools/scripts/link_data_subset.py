import os
import fire
from os import path as osp


def link_data_subset(kitti_dir, index_txt):
    indices = open(index_txt, 'r').read().splitlines()
    output_dir = os.path.join(kitti_dir, '../kitti_format_subset')
    output_label_dir = os.path.join(output_dir, 'label_2')
    output_calib_dir = os.path.join(output_dir, 'calib')
    output_velodyne_dir = os.path.join(output_dir, 'velodyne')
    if os.path.isdir(output_dir):
        print(f'output dir {output_dir} already exists !!')
        return
    os.mkdir(output_dir)
    os.mkdir(output_label_dir)
    os.symlink(osp.join(kitti_dir, 'calib'), output_calib_dir)
    os.symlink(osp.join(kitti_dir, 'velodyne'), output_velodyne_dir)
    for idx_name in indices:
        os.symlink(osp.join(kitti_dir, 'label_2', f'{idx_name}.txt'), osp.join(
            output_label_dir,  f'{idx_name}.txt'))


if __name__ == "__main__":
    fire.Fire()
