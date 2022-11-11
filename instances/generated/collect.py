import glob
import shutil

d = {}

with open('splist.txt') as f:
    for i in f:
        i = i.partition('.')[0]
        clones = glob.glob(f'feas/{i}*')
        if clones:
            d[i] = clones

n = len(d)
names = sorted(d.keys())
train_origs = names[:int(n*0.8)]
valid_origs = names[int(n*0.8):]

train = [f'origs/{i}.mps.gz' for i in train_origs]
valid = [f'origs/{i}.mps.gz' for i in valid_origs]

while len(train) < 128:
    for i in train_origs:
        if d[i]:
            train.append(d[i].pop(0))

while len(valid) < 48:
    for i in valid_origs:
        if d[i]:
            valid.append(d[i].pop(0))

print(len(train), len(valid))

for i in train:
    name = i.rpartition('/')[-1]
    shutil.copy(i, f'train/{name}')

for i in valid:
    name = i.rpartition('/')[-1]
    shutil.copy(i, f'valid/{name}')
