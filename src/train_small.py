import tensorflow as tf
import numpy as np
import json

def load_data(name, input_size=2048):
	content = []
	label = []
	id = []
	with open('../tmp/' + name + '.json', 'r') as f:
		cnt = 0;
		for line in f:
			cnt += 1
			#if (cnt > 5 * 64): break# and name != 'test'): break
			if (cnt == 1 or cnt % 811 == 0):
				print "Loading data_%s now. %d data loaded.\r" % (name, cnt),
			obj = json.loads(line)
			#print obj['content']
			content.append(obj['content'][0:input_size])
			#print content[len(content) - 1]
			if (name != 'test'):
				label.append(int(obj['label']))
			id.append(obj['id'])
	print "Loading data_%s now. %d data loaded." % (name, cnt - 1)
	return [np.array(content), np.array(label), id]

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.02)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.02, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride=[1, 1]):
	return tf.nn.conv2d(x, W, strides = [1, stride[0], stride[1], 1], padding = 'SAME')

def max_pool_2x1(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

def calc_auc(label, pred):
	s = zip(pred, label)
	s.sort(reverse = True)
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
	if n0 == 0 or n1 == 0:
		return 1
	for (p, l) in s:
		#print p, l
		if l == 1:
			ct1 += 1
		else:
			ct0 += 1
			res += (ct1 / (n1 + 0.0)) * (1. / (n0 + 0.0))
	return res

def train(epoch, batch_size, reg, voc_size, input_size, num_embed, filter_size,
		  num_filter, fc_size, num_fc):
	
	data = tf.placeholder(tf.int32, shape=[None, input_size])
	label = tf.placeholder(tf.float32, shape=[None])
	dropout = tf.placeholder(tf.float32)
	
	nConv_1 = len(num_filter[0])
	nConv_2 = len(num_filter[1])
	
	W_conv_1 = [0] * nConv_1
	b_conv_1 = [0] * nConv_1
	h_conv_1 = [0] * nConv_1
	h_pool_1 = [0] * nConv_1
	
	W_conv_2 = [0] * nConv_2
	b_conv_2 = [0] * nConv_2
	h_conv_2 = [0] * nConv_2
	h_pool_2 = [0] * nConv_2
	
	W_fc_1 = [0]
	b_fc_1 = [0]
	h_fc_1 = [0]
	h_drop_1 = [0]
	
	W_fc_2 = [0]
	b_fc_2 = [0]
	h_fc_2 = [0]
	h_drop_2 = [0]
	
	W_fc = [0] * num_fc[2]
	b_fc = [0] * num_fc[2]
	h_fc = [0] * num_fc[2]
	h_drop = [0] * num_fc[2]
	with tf.device('/gpu:2'):
		embed_weight = tf.Variable(tf.random_uniform([voc_size, num_embed], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embed_weight, data)
		embed_expanded = tf.expand_dims(embed, -1)
		
		#print embed_weight
		#print embed
		#print embed_expanded
		
		#embed_expanded = tf.Variable(tf.constant(1., shape=[batch_size, 2048, 7, 1]))
		
		size_1 = input_size
		
		for i in xrange(nConv_1):
			if i == 0:
				W_conv_1[i] = weight_variable([filter_size, num_embed, 1, num_filter[0][i]])
			else:
				W_conv_1[i] = weight_variable([filter_size, 1, num_filter[0][i - 1], num_filter[0][i]])
			b_conv_1[i] = bias_variable([num_filter[0][i]])
			if i == 0:
				#print embed_expanded
				h_conv_1[i] = tf.nn.relu(conv2d(embed_expanded, W_conv_1[i], stride=[1, num_embed])
										+ b_conv_1[i])
				#print h_conv[i]
			else:
				h_conv_1[i] = tf.nn.relu(conv2d(h_pool_1[i - 1], W_conv_1[i]) + b_conv_1[i])
			if i % 2 == 1 or i == 0:
				h_pool_1[i] = max_pool_2x1(h_conv_1[i])
				size_1 /= 2
			else:
				h_pool_1[i] = h_conv_1[i]
		
		size_2 = input_size
		
		for i in xrange(nConv_2):
			if i == 0:
				W_conv_2[i] = weight_variable([filter_size, num_embed, 1, num_filter[1][i]])
			else:
				W_conv_2[i] = weight_variable([filter_size, 1, num_filter[1][i - 1], num_filter[1][i]])
			b_conv_2[i] = bias_variable([num_filter[1][i]])
			if i == 0:
				#print embed_expanded
				h_conv_2[i] = tf.nn.relu(conv2d(embed_expanded, W_conv_2[i], stride=[1, num_embed])
										+ b_conv_2[i])
				#print h_conv[i]
			else:
				h_conv_2[i] = tf.nn.relu(conv2d(h_pool_2[i - 1], W_conv_2[i]) + b_conv_2[i])
			if i % 2 == 1 or i == 0:
				h_pool_2[i] = max_pool_2x1(h_conv_2[i])
				size_2 /= 2
			else:
				h_pool_2[i] = h_conv_2[i]
		
		#print h_conv[0]
		#h_pool[0] = tf.constant(1., shape=[batch_size, 1024, 7, 32])
		#print h_conv[0]
		#print h_pool[0]
		
		if nConv_1 >= 1:
			h_pool_flat_1 = tf.reshape(h_pool_1[nConv_1 - 1], [-1, size_1 * num_filter[0][nConv_1 - 1]])
		else:
			h_pool_flat_1 = tf.reshape(embed_expanded, [-1, input_size * num_embed])
		
		if nConv_2 >= 1:
			h_pool_flat_2 = tf.reshape(h_pool_2[nConv_2 - 1], [-1, size_2 * num_filter[1][nConv_2 - 1]])
		else:
			h_pool_flat_2 = tf.reshape(embed_expanded, [-1, input_size * num_embed])
		#print h_pool_flat
		#h_pool_flat = tf.constant(1., shape=[batch_size, 32768])
		
		if nConv_1 >= 1:
			W_fc_1[0] = weight_variable([size_1 * num_filter[0][nConv_1 - 1], fc_size])
		else:
			W_fc_1[0] = weight_variable([input_size * num_embed, fc_size])
		h_fc_1[0] = tf.nn.relu(tf.matmul(h_pool_flat_1, W_fc_1[0]) + b_fc_1[0])
		h_drop_1[0] = tf.nn.dropout(h_fc_1[0], dropout)
		
		if nConv_2 >= 1:
			W_fc_2[0] = weight_variable([size_2 * num_filter[1][nConv_2 - 1], fc_size])
		else:
			W_fc_2[0] = weight_variable([input_size * num_embed, fc_size])
		h_fc_2[0] = tf.nn.relu(tf.matmul(h_pool_flat_2, W_fc_2[0]) + b_fc_2[0])
		h_drop_2[0] = tf.nn.dropout(h_fc_2[0], dropout)
		
		for i in xrange(num_fc[2]):
			if i == 0:
				W_1 = weight_variable([fc_size, fc_size])
				W_2 = weight_variable([fc_size, fc_size])
				b_1 = bias_variable([fc_size])
				b_2 = bias_variable([fc_size])
			else:
				W_fc[i] = weight_variable([fc_size, fc_size])
				b_fc[i] = bias_variable([fc_size])
			if i == 0:
				h_fc[i] = tf.add(tf.nn.relu(tf.matmul(h_drop_1[0], W_1) + b_1),
				tf.nn.relu(tf.matmul(h_drop_2[0], W_2) + b_2))
			else:
				h_fc[i] = tf.nn.relu(tf.matmul(h_fc[i - 1], W_fc[i]) + b_fc[i])
			h_drop[i] = tf.nn.dropout(h_fc[i], dropout)
			#bn?
		W_final = weight_variable([fc_size, 2])
		b_final = bias_variable([2])
		pred = tf.nn.softmax(tf.matmul(h_drop[num_fc[2] - 1], W_final) + b_final)
		#print '--------------'
		#print pred[:, 1]
		#print label
		#print '--------------'
		
		weight_loss = tf.Variable(tf.constant(0.0))
		for i in xrange(nConv_1):
			weight_loss += reg * tf.nn.l2_loss(W_conv_1[i])
		for i in xrange(nConv_2):
			weight_loss += reg * tf.nn.l2_loss(W_conv_2[i])
		for i in xrange(num_fc[0]):
			weight_loss += reg * tf.nn.l2_loss(W_fc_1[i])
		for i in xrange(num_fc[1]):
			weight_loss += reg * tf.nn.l2_loss(W_fc_2[i])
		for i in xrange(1, num_fc[2]):
			weight_loss += reg * tf.nn.l2_loss(W_fc[i])
		weight_loss += reg * (tf.nn.l2_loss(W_fc[i]) + tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2))
		
		loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred[:, 1], label)))) + reg * weight_loss
		#auc, update_op = tf.contrib.metrics.streaming_auc(pred[:, 1], label + 1.0)
		
		train_step = tf.train.AdamOptimizer(2e-4).minimize(loss)

	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	len_train = data_train[0].shape[0]
	len_val = data_val[0].shape[0]
	len_test = data_test[0].shape[0]
	nIter = len_train / batch_size
	print "Training..."
	for i in xrange(epoch):
		cnt = 0
		for j in xrange(0, len_train, batch_size):
			batch_x = data_train[0][j:j+batch_size]
			batch_y = data_train[1][j:j+batch_size]
			cnt += 1
			if cnt == 5:
				cnt = 0
				train_y = pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1]
				train_auc = calc_auc(batch_y, train_y)
				print "train_auc = %g" % train_auc
			for k in xrange(len(batch_x)):
				if batch_y[k] == 1:
					batch_x = np.append(batch_x, [batch_x[k] for t in xrange(10)], axis = 0)
					batch_y = np.append(batch_y, [1 for t in xrange(10)], axis = 0)
			#print batch_x.shape
			#print batch_y.shape
			train_step.run(feed_dict = {data: batch_x, label: batch_y, dropout: 0.5})
		val_y = np.array([])
		#print len_train, len_val, len_test
		for j in xrange(0, len_val, batch_size):
			batch_x = data_val[0][j:j+batch_size]
			tmp = pred.eval(feed_dict = {data: batch_x, dropout: 1.0})
			val_y = np.append(val_y, tmp[:, 1])
		#print tmp[:, 1]
		#print val_y
		val_auc = calc_auc(data_val[1], val_y)
		#for (p, q) in zip(data_val[1], val_y):
		#	print p, q
		#print '============='
		#print train_auc, val_auc
		print("step %d, val auc %g"%(i, val_auc))
		test_y = np.array([])
		for j in xrange(0, len_test, batch_size):
			batch_x = data_test[0][j:j+batch_size]
			test_y = np.append(test_y, pred.eval(feed_dict = {data: batch_x, dropout: 1.0})[:, 1])
		f_out = file('../data/test%d.csv' % i, 'w')
		f_out.write('id,pred\n')
		for (i, j) in zip(data_test[2], test_y):
			f_out.write('%s,%.30f\n'%(i, j))
		f_out.close()
		print "answer generated"

if __name__ == '__main__':
	
	#dh.preprocess(mode=mode)
	#dh.statistic()
	
	input_size = 384
	num_embed = 96
	filter_size = 3
	num_filter = [[256], [128, 128, 256, 256]]
	fc_size = 150
	num_fc = [1, 1, 2]
	
	epoch = 5
	batch_size = 25
	reg = 1e-3
	
	# loading data
	print 'Loading data...'
	voc_size = 1331065
	data_train = load_data('train', input_size=input_size)
	data_val = load_data('val', input_size=input_size)
	data_test = load_data('test', input_size=input_size)
	
	print data_train[0].shape
	print data_val[0].shape
	print data_test[0].shape
	
	train(epoch, batch_size, reg, voc_size, input_size, num_embed, filter_size,
		  num_filter, fc_size, num_fc)
