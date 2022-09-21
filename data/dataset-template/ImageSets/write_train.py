import os

label_dir = '../training/label_2'

train = [int(x.split('.')[0]) for x in os.listdir(label_dir)]
train.sort()
val = [x for x in train if x %10 ==0]
# val_len = 33770 - 30001
# val = [i for i in range(90002, 90002+val_len)]
# train = [i for i in files if i not in val]
with open('train.txt', 'w') as out:
    for x in train:
        out.write(f'{x:06}' + "\n")
with open('val.txt', 'w') as out:
    for x in val:
        out.write(f'{x:06}' + "\n")
