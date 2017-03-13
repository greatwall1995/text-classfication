import tensorflow as tf
import numpy as np
import data_helpers
import json
import sys

def get_voc(mode):
	if mode == 'char':
		name = 'dict1'
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
			#if (cnt > 20 * 64): break# and name != 'test'): break
			if (cnt == 1 or cnt % 811 == 0):
				print "Loading data_%s now. %d data loaded.\r" % (name, cnt),
			obj = json.loads(line)
			#print obj['content']
			content.append(translate(obj['content'], voc, input_size, mode = 'char'))
			#print content[len(content) - 1]
			if (name != 'test'):
				label.append(int(obj['label']))
			id.append(obj['id'])
	print "Loading data_%s now. %d data loaded." % (name, cnt - 1)
	return [np.array(content), np.array(label), id]

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride=[1, 1]):
	return tf.nn.conv2d(x, W, strides = [1, stride[0], stride[1], 1], padding = 'SAME')

def max_pool_2x1(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

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
		#print p, l
		if l == 1:
			ct1 += 1
		else:
			ct0 += 1
			res += (ct1 / (n1 + 0.0)) * (1. / (n0 + 0.0))
	#print '=========================='
	return res

def train(epoch, batch_size, reg, voc_size, input_size, num_embed, filter_size, num_filter, fc_size):
	
	data = tf.placeholder(tf.int32, shape=[None, input_size])
	label = tf.placeholder(tf.float32, shape=[None])
	dropout = tf.placeholder(tf.float32)
	
	nConv = len(filter_size)
	nFc = len(fc_size)
	
	W_conv = [0] * nConv
	b_conv = [0] * nConv
	h_conv = [0] * nConv
	h_pool = [0] * nConv
	
	W_fc = [0] * nFc
	b_fc = [0] * nFc
	h_fc = [0] * nFc
	h_drop = [0] * nFc
	
	embed_weight = tf.Variable(tf.random_uniform([voc_size, num_embed], -1.0, 1.0))
	embed = tf.nn.embedding_lookup(embed_weight, data)
	embed_expanded = tf.expand_dims(embed, -1)
	
	#print embed_weight
	#print embed
	#print embed_expanded
	
	#embed_expanded = tf.Variable(tf.constant(1., shape=[batch_size, 2048, 7, 1]))
	
	size = input_size
	
	for i in xrange(nConv):
		if i == 0:
			W_conv[i] = weight_variable([filter_size[i], num_embed, 1, num_filter[i]])
		else:
			W_conv[i] = weight_variable([filter_size[i], 1, num_filter[i - 1], num_filter[i]])
		b_conv[i] = bias_variable([num_filter[i]])
		if i == 0:
			#print embed_expanded
			h_conv[i] = tf.nn.relu(conv2d(embed_expanded, W_conv[i], stride=[1, num_embed]) + b_conv[i])
			#print h_conv[i]
		else:
			h_conv[i] = tf.nn.relu(conv2d(h_pool[i - 1], W_conv[i]) + b_conv[i])
		h_pool[i] = max_pool_2x1(h_conv[i])
		size /= 2
	#print h_conv[0]
	#h_pool[0] = tf.constant(1., shape=[batch_size, 1024, 7, 32])
	#print h_conv[0]
	#print h_pool[0]
	
	if nConv >= 1:
		h_pool_flat = tf.reshape(h_pool[nConv - 1], [-1, size * num_filter[nConv - 1]])
	else:
		h_pool_flat = tf.reshape(embed_expanded, [-1, input_size * num_embed])
	#print h_pool_flat
	#h_pool_flat = tf.constant(1., shape=[batch_size, 32768])
	
	for i in xrange(nFc):
		if i == 0:
			if nConv >= 1:
				W_fc[i] = weight_variable([size * num_filter[nConv - 1], fc_size[i]])
			else:
				W_fc[i] = weight_variable([input_size * num_embed, fc_size[i]])
		else:
			W_fc[i] = weight_variable([fc_size[i - 1], fc_size[i]])
		b_fc[i] = bias_variable([fc_size[i]])
		if i == 0:
			h_fc[i] = tf.nn.relu(tf.matmul(h_pool_flat, W_fc[i]) + b_fc[i])
		else:
			h_fc[i] = tf.nn.relu(tf.matmul(h_fc[i - 1], W_fc[i]) + b_fc[i])
		h_drop[i] = tf.nn.dropout(h_fc[i], dropout)
		#bn?
	W_final = weight_variable([fc_size[nFc - 1], 2])
	b_final = bias_variable([2])
	pred = tf.nn.softmax(tf.matmul(h_drop[nFc - 1], W_final) + b_final)
	weight_loss = tf.Variable(tf.constant(0.0))
	for i in xrange(nConv):
		weight_loss += reg * tf.nn.l2_loss(W_conv[i])
	for i in xrange(nFc):
		weight_loss += reg * tf.nn.l2_loss(W_fc[i])
	
	loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred[:, 1], label)))) + reg * weight_loss
	
	train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)
	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	saver.restore(sess, 'my-model-1')
	val_y = np.array([])
	len_val = len(data_val[0])
	len_test = len(data_test[0])
	cnt = 0
	for j in xrange(0, len_val, batch_size):
		batch_x = data_val[0][j:j+batch_size]
		batch_y = data_val[1][j:j+batch_size]
		cnt += 1
		if cnt == 20:
			cnt = 0
			print j, len_val
		#train_auc = calc_auc(batch_y, pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1])
		#print pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1]
		val_y = np.append(val_y, pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1])
		#print j
		#sys.stdout.flush()
	#print data_val[1].T
	#print test_y
	val_auc = calc_auc(data_val[1], val_y)
	print("val auc %g"%(val_auc))
	#saver.save(sess, 'my-model', global_step=i)
	test_y = np.array([])
	for j in xrange(0, len_test, batch_size):
		batch_x = data_test[0][j:j+batch_size]
		test_y = np.append(test_y, pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1])
	f_out = file('../data/test.csv', 'w')
	f_out.write('id, pred\n')
	for (i, j) in zip(data_test[2], test_y):
		#print i, j
		f_out.write('%s,%.200f\n'%(i, j))
	f_out.close()

if __name__ == '__main__':
	
	mode = 'char'
	
	input_size = 1024
	num_embed = 256
	filter_size = [5, 3, 3]
	num_filter = [128, 256, 384]
	fc_size = [100, 100]
	
	epoch = 20
	batch_size = 64
	reg = 1e-3
	
	# loading data
	print 'Loading data...'
	voc = get_voc(mode)
	voc_size = len(voc)
	#data_train = load_data('train', voc, input_size=input_size, mode=mode)
	data_val = load_data('val', voc, input_size=input_size, mode=mode)
	data_test = load_data('test', voc, input_size=input_size, mode=mode)
	
	#print data_train[0].shape
	#print data_val[0].shape
	#print data_test[0].shape
	
	train(epoch, batch_size, reg, voc_size, input_size, num_embed, filter_size, num_filter, fc_size)
