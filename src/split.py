# split train.json into trainning data (90%) and validation data (10%).

import json

train_f = file('../data/train.json', 'w')
val_f = file('../data/val.json', 'w')

cnt = 0

with open('../data/org.json', 'r') as f:
	for line in f:
		cnt += 1
		if cnt % 10 != 0:
			train_f.write(line)
		else:
			val_f.write(line)

print 'cnt = ', cnt