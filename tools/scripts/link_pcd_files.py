import os
import fire


def link_pcd_files(src_pcd_dir, dst_pcd_dir, ann_dir):
    indices = [int(x.split('.')[0]) for x in os.listdir(ann_dir)]
    for index in indices:
        src_pcd_path = os.path.realpath(
            os.path.join(src_pcd_dir, f'{index:06}.pcd'))
        dst_pcd_path = os.path.join(dst_pcd_dir, f'{index:06}.pcd')
        print(f'link from {src_pcd_path} to {dst_pcd_path}')
        os.symlink(src_pcd_path, dst_pcd_path)


if __name__ == "__main__":
    fire.Fire()
