import os
import fire


def write_index_file(index_dir, label_dir):
    train = [int(x.split('.')[0]) for x in os.listdir(label_dir)
             if len(open(os.path.join(label_dir, x), 'r').readlines()) != 0]
    train.sort()
    val = train[:]
    with open(os.path.join(index_dir, 'train.txt'), 'w') as out:
        for x in train:
            out.write(f'{x:06}' + "\n")
    with open(os.path.join(index_dir, 'val.txt'), 'w') as out:
        for x in val:
            out.write(f'{x:06}' + "\n")


if __name__ == '__main__':
    fire.Fire()
