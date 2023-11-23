from os import path as osp

import os
import fire
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
# dirs = ['calib', 'label_2', 'velodyne']

# for dir in dirs:


def analyze_dataset(dir):
    files = os.listdir(dir)
    files.sort()
    obj_map = defaultdict(list)
    for file in tqdm(files):
        anno = json.load(open(osp.join(dir, file), 'r'))
        assert len(anno['objects']) == len(anno['figures'])
        assert [x['key'] for x in anno['objects']] == [x['objectKey']
                                                       for x in anno['figures']]
        for obj, fig in zip(anno['objects'], anno['figures']):
            if obj['classTitle'] == 'None':
                continue
            obj_map[obj['classTitle']].append(fig)
    names = list(obj_map.keys())
    names.sort(key=lambda x: len(obj_map[x]))
    total_cnt = len([x for y in obj_map.values() for x in y])
    print(f'Total objects: {total_cnt}')
    for name in names:
        fig_sizes = np.array([(fig['geometry']['dimensions']['y'], fig['geometry']
                               ['dimensions']['x'], fig['geometry']['dimensions']['z']) for fig in obj_map[name]])
        avg_sizes = sum(fig_sizes) / fig_sizes.shape[0]
        stddev_sizes = np.std(fig_sizes, axis=0)
        print(name, len(obj_map[name]))
        print(np.round(avg_sizes, 2))
        print(np.round(stddev_sizes, 2))


if __name__ == '__main__':
    fire.Fire()
