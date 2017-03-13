import json
import re
import pypinyin

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
			if (obj['label'] == 'True' or obj['label'] == 'true'):
				print 'True'
				obj['label'] = 1
			elif (obj['label'] == 'False' or obj['label'] == 'false'):
				print 'False'
				obj['label'] = 0
			line = json.dumps(obj)
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

def get_dict(name):
	ret = set()
	with open('../data/' + name + '.txt', 'r') as f:
		for line in f:
			#print line[2:-2].decode('unicode_escape')
			ret.add(line[2:-2].decode('unicode_escape'))
	return ret

def modify1(s):
	'''
	Character based sentence.
	'''
	dic = get_dict('dict1')
	ret = strQ2B(ret)
	l = len(ret);
	tmp = ''
	flag = False
	for i in xrange(l):
		if flag:
			tmp += ' '
		if (u'A' <= ret[i] and ret[i] <= u'Z'):
			tmp += unichr(ord(ret[i]) - ord(u'A') + ord(u'a'));
			flag = False
		elif (ret[i] == u'\u3010'):
			tmp += u'['
			flag = False
		elif (ret[i] == u'\u3011'):
			tmp += u']'
			flag = False
		elif (ret[i] == u'\u2571'):
			tmp += u'/'
			flag = False
		elif (ret[i] == u'\u2014'):
			tmp += u'-'
			flag = False
		elif (ret[i] == u'\u00a0' or ret[i] == u'\n' or ret[i] == u'\r' or ret[i] == u'\t'):
			tmp += u' '
			flag = False
		elif ret[i] in dic:
			tmp += ret[i]
			flag = False
		else:
			t = pypinyin.pinyin(ret[i])
			if (t[0][0] != ret[i] and t[0][0].find(u'\u0144') == -1):
				if not flag and i != 0:
					tmp += ' '
				tmp += t[0][0]
				flag = True
	return tmp

def prep(name, input_size):
	'''
	Dealing with raw data.
	'''
	
	f_out = file('../tmp/' + name + '.json', 'w')
	
	cnt = 0;
	
	with open('../data/' + name + '.json', 'r') as f_in:
		for line in f_in:
			
			cnt += 1
			if (cnt % 1000 == 0):
				print cnt
			
			obj = json.loads(line)
			content = modify1(obj['content'])
			title = modify1(obj['title'])
			res = ''
			pos = 0
			cLen = len(content)
			tLen = len(title)
			
			res = title
			pos = tLen
			
			while pos < input_size:
				res += ' '
				pos += 1
				if (pos + cLen <= input_size):
					res += content
					pos += cLen
				else:
					idx = 0
					while pos < input_size:
						res += content[idx]
						pos += 1
						idx += 1
					break
			#print obj['title']
			#print obj['content']
			#print res
			
			obj['content'] = res
			obj.pop('title')
			f_out.write(json.dumps(obj) + '\n')
		
	f_out.close()

def preprocess(input_size=2048):
	
	print 'Preprocessing...'
	prep("train", input_size)
	prep("val", input_size)
	prep("test", input_size)
	print 'Preprocessing done.'

def stat(name, dic):
	
	with open('../tmp/' + name + '.json', 'r') as f:
		for line in f:
			cont = json.loads(line)['content']
			for c in cont:
				dic[c] = dic.get(c, 0) + 1

def statistic():
	dic = {}
	
	stat("train", dic)
	stat("val", dic)
	stat("test", dic)
	
	stat_file = file('../tmp/stat.txt', 'w')
	for c in dic:
		obj = {'char': c, 'num': dic[c]}
		stat_file.write(json.dumps(obj) + '\n')
	stat_file.close()