import os
import fire
import json
import shutil
# dirs = ['calib', 'label_2', 'velodyne']

# for dir in dirs:


def reindex_annotations(src_ann_dir, dst_ann_dir, index_map, map_reverse=False):
    index_map = json.load(open(index_map, 'r'))
    index_map = {int(x[0].split('.')[0]): int(x[1].split('.')[0])
                 for x in index_map}
    if map_reverse:
        index_map = {v: k for k, v in index_map.items()}
    src_files = os.listdir(src_ann_dir)
    src_files.sort()
    for src_file in src_files:
        src_index = int(src_file.split('.')[0])
        dst_index = index_map[src_index % 100000]
        dst_file_name = src_file.replace(f'{src_index:06}', f'{dst_index:06}')

        dst_path = os.path.join(dst_ann_dir, dst_file_name)
        print(f'copy src_file {src_file} to {dst_path}')
        shutil.copyfile(os.path.join(src_ann_dir, src_file), dst_path)


if __name__ == '__main__':
    fire.Fire()
