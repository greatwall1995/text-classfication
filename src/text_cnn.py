import mxnet as mx
import numpy as np
import time
import json
import data_helpers

def translate(content, mode='char'):
	dic = data_helpers.get_dict('dict1')
	voc = {}
	for (i, ch) in enumerate(dic):
		voc[ch] = i
	l = len(content)
	res = []
	for ch in content:
		res.append(voc[ch])
	return res

def load_data(name, mode='char'):
	content = []
	label = []
	id = []
	with open('../tmp/' + name + '.json', 'r') as f:
		cnt = 0;
		for line in f:
			cnt += 1
			if (cnt == 10): break
			obj = json.loads(line)
			
			content.append(translate(obj['content'], mode = 'char'))

			if (name != 'test'):
				label.append(obj['label'])
			id.append(obj['id'])
	
	return [content, label, id]
	
def main():
	
	mode = 'char'	# mode = 'char' or 'word'
	input_size = 2048
	
	#data_helpers.split()
	#data_helpers.preprocess(input_size)
	#data_helpers.statistic()
	
	# loading data
	data_train = load_data('train', mode=mode)
	data_val = load_data('val', mode=mode)
	data_test = load_data('test', mode=mode)
	

if __name__ == '__main__':
	main()