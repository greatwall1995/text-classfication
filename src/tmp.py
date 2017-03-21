import json

f_out = file('../tmp/dict2.txt', 'w')

with open('../data/dict2.txt', 'r') as f:
	for line in f:
		obj = {'char': line}
		f_out.write(json.dumps(obj) + '\n')
		