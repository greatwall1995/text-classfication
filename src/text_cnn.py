import mxnet as mx
import numpy as np
import time
import json
import data_helpers
import logging

logging.basicConfig(level=logging.DEBUG)

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
			if (cnt > 5 * 64): break
			if (cnt % 1000 == 0):
				print "Loading data_%s now. %d data loaded.\r" % (name, cnt),
			obj = json.loads(line)
			#print obj['content']
			content.append(translate(obj['content'], voc, input_size, mode = 'char'))
			#print content[len(content) - 1]
			if (name != 'test'):
				label.append(int(obj['label']))
			id.append(obj['id'])
	print ''
	return [content, label, id]

def make_text_cnn(batch_size, input_size, voc_size, num_embed, filter_size,
				  num_filter, fc_size, dropout):
	input_x = mx.sym.Variable('data')
	embed_layer = mx.sym.Embedding(data=input_x, input_dim=voc_size, output_dim=num_embed,
								  name='embed')
	cnn = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, input_size, num_embed))
	t = num_embed
	for i in xrange(len(filter_size)):
		fs = filter_size[i]
		nf = num_filter[i]
		cnn = mx.sym.Convolution(name='conv%d' % i, data=cnn, kernel=(fs, t), num_filter=nf)
		cnn = mx.sym.Activation(data=cnn, act_type='relu')
		#cnn = mx.sym.BatchNorm(data=cnn)
		cnn = mx.sym.Pooling(data=cnn, pool_type='max', kernel=(fs, 1), stride=(1,1))
		if dropout > 0.0:
			cnn = mx.sym.Dropout(data=cnn, p=dropout)
		t = 1
	fc = cnn
	for i in xrange(len(fc_size)):
		fc = mx.sym.FullyConnected(name='fc%d' % i, data=fc, num_hidden=fc_size[i])
  		fc = mx.sym.Activation(data=fc, act_type='relu')
    	#fc = mx.sym.BatchNorm(data=fc)
	fc = mx.sym.FullyConnected(name='final', data=fc, num_hidden=1)
	sm = mx.sym.LogisticRegressionOutput(data=fc)
	return sm

def setup_cnn_model(ctx, batch_size, input_size, voc_size, num_embed, filter_size,
					num_filter, fc_size, dropout=0.0):

	return make_text_cnn(batch_size, input_size, voc_size, num_embed, filter_size,
						num_filter, fc_size, dropout)


def calc_auc(label, pred):
	s = zip(pred, label)
	s.sort(reverse = True)
	#print pred
	#print label
	n0 = 0
	n1 = 0
	for l in label:
		if l == 0:
			n0 += 1
		else:
			n1 += 1
	res = 0.0
	ct0 = 0
	ct1 = 0
	for (p, l) in s:
		if l == 1:
			ct1 += 1
		else:
			ct0 += 1
			res += (ct1 / (n1 + 0.0)) * (1. / (n0 + 0.0))
	return res

class eval_call(object):
	def __call__(self, param):
		print "eval_call()"
		print param
		print ""

def epoch_call():
	def _callback(iter_no, sym, arg, aux):
		print "epoch_call()"
		print "iter_no = ", iter_no
		print "sym = ", sym
		print "arg = ", arg
		print "aux = ", aux
		print ""
	return _callback

def train_cnn(ctx, cnn_model, data_train, data_val, data_test, batch_size):
	print 'Trainning model...'	
	epoch = 5
	learning_rate = 1e-1
	reg = 5e-4
	
	model = mx.model.FeedForward(
		symbol=cnn_model,
		ctx=ctx,
		optimizer='adam',
		learning_rate=learning_rate,
		numpy_batch_size=batch_size,
		wd=reg,
		num_epoch=1
	)
	
	print np.array(data_train[0]).shape
	print np.array(data_train[1]).shape
	print np.array(data_val[0]).shape
	print np.array(data_val[1]).shape
	
	print batch_size
	
	train = mx.io.NDArrayIter(np.array(data_train[0]), np.array(data_train[1]),
							  batch_size=batch_size, label_name='logisticregressionoutput0_label')#, shuffle=True)
	print batch_size
	val = mx.io.NDArrayIter(np.array(data_val[0]), np.array(data_val[1]),
							batch_size=batch_size)
	print batch_size
	test = mx.io.NDArrayIter(np.array(data_test[0]), np.array(data_test[1]),
							 batch_size=batch_size)
	
	best_model = None
	best_auc = -1
	
	#model.fit(X=train, eval_data=val,
	#		  batch_end_callback = mx.callback.Speedometer(batch_size, 1),
	#		  epoch_end_callback = epoch_call(),
	#		  eval_end_callback = eval_call())
	
	for i in xrange(epoch):
		model.fit(X=train, batch_end_callback = mx.callback.Speedometer(batch_size, 1))
		prob = model.predict(np.array(data_val[0]))#, num_batch=batch_size)
		print prob
		auc = calc_auc(data_val[1], prob)
		print "Iter %d, AUC = %f" % (i, auc)
		model.save('model', i)
		if auc > best_auc:
			best_auc = auc
			best_model = model
	
	print "Best AUC = %f" % best_auc
	best_model.save('best_model', 0)
	prob = best_model.predict(test, num_batch=batch_size)
	f_out1 = file('../data/test.txt', 'w')
	print prob
	f_out1.close()
	f_out = file('../data/test.csv', 'w')
	print 'id, pred'
	for (i, j) in xrange(zip(data_text, prob)):
		print i,j
	f_out.close()

def main():
	
	mode = 'char'	# mode = 'char' or 'word'
	input_size = 2048
	num_embed = 32#128
	batch_size = 64
	filter_size = [5, 3]
	num_filter = [16, 18]#[128, 256]
	fc_size = [50]#[200]
	dropout = 0.5
	
	voc = get_voc('dict1')
	
	voc_size = len(voc)
	
	#data_helpers.split()
	#data_helpers.preprocess()
	#data_helpers.statistic()
	
	# loading data
	print 'Loading data...'
	data_train = load_data('train', voc, mode=mode)
	data_val = load_data('val', voc, mode=mode)
	data_test = load_data('test', voc, mode=mode)
	
	#print data_train[0]
	#print '------------------'
	#print data_val[0]
	#print '------------------'
	
	print 'train shape', len(data_train[0])
	print 'val shape', len(data_val[0])
	print 'test shape', len(data_test[0])
	
	cnn_model = setup_cnn_model(mx.cpu(0), batch_size, input_size, voc_size, num_embed,
								filter_size, num_filter, fc_size, dropout)
	train_cnn(mx.cpu(0), cnn_model, data_train, data_val, data_test, batch_size)

if __name__ == '__main__':
	main()