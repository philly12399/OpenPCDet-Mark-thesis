import os
import fire


def append_links(kitti_dir, src_kitti_dir):
    label_dir = os.path.join(kitti_dir, 'label_2')
    train = [int(x.split('.')[0]) for x in os.listdir(label_dir)]


if __name__ == '__main__':
    fire.Fire()
