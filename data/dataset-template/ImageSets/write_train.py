import os
file_path = os.path.dirname(os.path.realpath(__file__))
label_dir = os.path.join(file_path, '../training/label_2')

train = [int(x.split('.')[0]) for x in os.listdir(label_dir)]
train.sort()
val = [x for x in train if x %10 ==0]
with open(os.path.join(file_path, 'train.txt'), 'w') as out:
    for x in train:
        out.write(f'{x:06}' + "\n")
with open(os.path.join(file_path, 'val.txt'), 'w') as out:
    for x in val:
        out.write(f'{x:06}' + "\n")
