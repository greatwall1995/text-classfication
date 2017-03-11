import mxnet as mx
import numpy as np
import time
import json
import data_helpers

def get_voc(name):
	dic = data_helpers.get_dict(name)
	voc = {}
	for (i, ch) in enumerate(dic):
		voc[ch] = i
	return voc

def translate(content, voc, input_size, mode='char'):
	l = len(content)
	res = []
	for i in xrange(input_size):
		res.append(voc[content[i]])
	return res

def load_data(name, voc, input_size=2048, mode='char'):
	content = []
	label = []
	id = []
	with open('../tmp/' + name + '.json', 'r') as f:
		cnt = 0;
		for line in f:
			cnt += 1
			if (cnt % 1000 == 0):
				print "Loading data_%s now. %d data loaded.\r" % (name, cnt),
			obj = json.loads(line)
			#print obj['content']
			content.append(translate(obj['content'], voc, input_size, mode = 'char'))
			#print content[len(content) - 1]
			if (name != 'test'):
				label.append(obj['label'])
			id.append(obj['id'])
	print ''
	return [content, label, id]
	
def main():
	
	mode = 'char'	# mode = 'char' or 'word'
	input_size = 2048
	voc = get_voc('dict1')
	
	#data_helpers.split()
	#data_helpers.preprocess(input_size)
	#data_helpers.statistic()
	
	# loading data
	print 'Loading data...'
	data_train = load_data('train', voc, mode=mode)
	data_val = load_data('val', voc, mode=mode)
	data_test = load_data('test', voc, mode=mode)
	
	print 

if __name__ == '__main__':
	main()