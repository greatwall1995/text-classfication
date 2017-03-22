import json
import re
import jieba

def split():
	'''
	Splitting train.json into trainning data (90%) and validation data (10%).
	'''
	
	print 'Splitting data...'
	
	train_f = file('../data/train.json', 'w')
	val_f = file('../data/val.json', 'w')

	cnt = 0

	with open('../data/org.json', 'r') as f:
		for line in f:
			cnt += 1
			obj = json.loads(line)
			if (obj['label'] == True):
				print 'True'
				obj['label'] = 0
			elif (obj['label'] == False):
				print 'False'
				obj['label'] = 0
			line = json.dumps(obj) + '\n'
			if cnt % 10 != 0:
				train_f.write(line)
			else:
				val_f.write(line)
	train_f.close()
	val_f.close()
	print 'Number of samples:', cnt
	
def strQ2B(s):
	res = ""
	for ch in s:
		inside_code = ord(ch)
		if (inside_code == 12288):
			inside_code = 32
		elif (inside_code >= 65281 and inside_code <= 65374):
			inside_code -= 65248
		res += unichr(inside_code)
	return res

#def get_dict():
	#ret = set()
	#with open('../data/dict.txt', 'r') as f:
	#	for line in f:
	#		ret.add(line[2:-2].decode('unicode_escape'))
	#return ret

def modify(s, dic):
	'''
	Character based sentence.
	'''
	#dic = get_dict('dict')
	ret = strQ2B(s)
	ret = jieba.lcut(ret)
	tmp = []
	l = len(ret);
	for i in xrange(l):
		if (u'A' <= ret[i] and ret[i] <= u'Z'):
			tmp.append(ret[i].lower())
		elif (ret[i] == u'\u3010'):
			tmp.append(u'[')
		elif (ret[i] == u'\u3011'):
			tmp.append(u']')
		elif (ret[i] == u'\u2571'):
			tmp.append(u'/')
		elif (ret[i] == u'\u2014'):
			tmp.append(u'-')
		elif (ret[i] == u'\u00a0' or ret[i] == u'\n' or ret[i] == u'\r' or ret[i] == u'\t'):
			tmp.append(u' ')
		else:
			tmp.append(ret[i])
	ret = tmp
	for i in xrange(l):
		word = ret[i]
		#print ret
		if (dic.has_key(word)):
			ret[i] = dic[word]
		else:
			ret[i] = len(dic)
			dic[word] = len(dic)
	return ret

def prep(name, input_size, dic):
	'''
	Dealing with raw data.
	'''
	
	f_out = file('../tmp/' + name + '.json', 'w')
	
	cnt = 0
	
	with open('../data/' + name + '.json', 'r') as f_in:
		for line in f_in:
			
			cnt += 1
			if cnt % 1000 == 0:
				print 'solved %d' % cnt
			
			obj = json.loads(line)
			obj['content'] = re.sub(u'<[ -~]*>', u'', obj['content'])
			obj['title'] = re.sub(u'<[ -~]*>', u'', obj['title'])
			#print obj['title']
			#print obj['content']
			content = modify(obj['content'], dic)
			title = modify(obj['title'], dic)
			res = []
			pos = 0
			cLen = len(content)
			tLen = len(title)
			
			res = title
			pos = tLen
			
			while pos < input_size:
				if (cLen == 0):
					res.append(dic[' '])
					pos += 1
				elif (pos + cLen <= input_size):
					res.extend(content)
					pos += cLen
				else:
					idx = 0
					while pos < input_size:
						res.append(content[idx])
						pos += 1
						idx += 1
					break
			#print obj['title']
			#print obj['content']
			#print res
			#print '---------------------------------'
			
			obj['content'] = res
			obj.pop('title')
			f_out.write(json.dumps(obj) + '\n')
		
	f_out.close()

def preprocess(input_size=2048):
	
	dic = {}
	print 'Preprocessing...'
	prep("train", input_size, dic)
	prep("val", input_size, dic)
	prep("test", input_size, dic)
	print 'Preprocessing done.'
	print 'Outputing dictionary'
	with file('../data/dict.txt', 'w') as f:
		for word in dic:
			f.write(word.encode('utf-8'))
			f.write(" %d\n" % dic[word])
	print 'Done'
