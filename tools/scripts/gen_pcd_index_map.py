import os
import numpy as np
import fire
import json


def max_pcd_timestamp(filepath):
    times = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[11:]:
            linestr = line.split(" ")
            # linestr_convert = list(map(float, linestr[:3]))
            # linestr_convert.append(0)
            # lidar.append(linestr_convert)
            if linestr[5] != '1':
                continue
            times.append(int(linestr[4]))
    return max(times)


def gen_pcd_index_map(src_pcd_dir, dst_pcd_dir):
    src_pcd_names = os.listdir(src_pcd_dir)
    src_pcd_names.sort()

    dst_pcd_names = os.listdir(dst_pcd_dir)
    dst_pcd_names.sort()

    dst_pcd_idx = 0
    dst_pcd_time = max_pcd_timestamp(os.path.join(
        dst_pcd_dir, dst_pcd_names[dst_pcd_idx]))

    index_map = []

    for src_pcd_name in src_pcd_names:
        src_pcd_path = os.path.join(src_pcd_dir, src_pcd_name)
        print(f'Processing src pcd file {src_pcd_name} ...')
        src_pcd_time = max_pcd_timestamp(src_pcd_path)

        while abs(src_pcd_time - dst_pcd_time) > 0.01 * 1_000_000_000:
            dst_pcd_idx += 1
            dst_pcd_time = max_pcd_timestamp(os.path.join(
                dst_pcd_dir, dst_pcd_names[dst_pcd_idx]))

        print(
            f'{src_pcd_name} matches {dst_pcd_names[dst_pcd_idx]} with diff {abs(src_pcd_time - dst_pcd_time)}')
        index_map.append((src_pcd_name, dst_pcd_names[dst_pcd_idx]))

        # print(src_pcd_names[:10], dst_pcd_names[:10])

    with open('index_map.json', 'w') as jsonfile:
        json.dump(index_map, jsonfile)


if __name__ == "__main__":
    fire.Fire()
