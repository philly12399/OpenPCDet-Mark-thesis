import os
import numpy as np
import fire


def read_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[11:]:
            linestr = line.split(" ")
            linestr_convert = list(map(float, linestr[:3]))
            linestr_convert.append(0)
            lidar.append(linestr_convert)
    return np.array(lidar)


def convert(pcdfolder, binfolder, convert_number):
    file_list = os.listdir(pcdfolder)
    file_list.sort()
    count = 0
    if os.path.exists(binfolder):
        pass
    else:
        os.makedirs(binfolder)
    for file in file_list:
        print(count)
        (filename, extension) = os.path.splitext(file)
        if extension == ".pcd" and count < convert_number:
            velodyne_file = os.path.join(pcdfolder, filename) + '.pcd'
            pl = read_pcd(velodyne_file)
            pl = pl.reshape(-1, 4).astype(np.float32)
            velodyne_file_new = os.path.join(binfolder, filename) + '.bin'
            pl.tofile(velodyne_file_new)
            count += 1


if __name__ == "__main__":
    fire.Fire()
