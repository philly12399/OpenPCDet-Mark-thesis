import os
import fire
# dirs = ['calib', 'label_2', 'velodyne']

# for dir in dirs:


def append_device_id(dir, id):
    files = os.listdir(dir)
    files.sort()
    for file in files:
        old_index = int(file.split('.')[0]) % 100000
        new_index = old_index + 100000 * id
        new_name = file.replace(file.split('.')[0], f'{new_index:>06}')
        print(file, new_name)
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name))


if __name__ == '__main__':
    fire.Fire()
