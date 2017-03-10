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

def modify(s):
	ret = re.sub(r'<[^<>]*>', '', s)
	ret = strQ2B(ret)
	l = len(ret);
	tmp = ''
	for i in xrange(l):
		if (u'A' <= ret[i] and ret[i] <= u'Z'):
			tmp += unichr(ord(ret[i]) - ord(u'A') + ord(u'a'));
		elif (ret[i] == u'\u3010'):
			tmp += u'['
		elif (ret[i] == u'\u3011'):
			tmp += u']'
		elif (ret[i] == u'\u2571'):
			tmp += u'/'
		elif (ret[i] == u'\u2014'):
			tmp += u'-'
		elif (ret[i] == u'\u00a0' or ret[i] == u'\n' or ret[i] == u'\r' or ret[i] == u'\t'):
			tmp += u' '
		else:
			tmp += ret[i]
	ret = pypinyin.pinyin(tmp)
	return ret

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
			content = modify(obj['content'])
			title = modify(obj['title'])
			res = ''
			pos = 0
			cLen = len(content)
			tLen = len(title)
			
			for i in xrange(tLen):
				s = title[i][0]
				res += s + ' '
				pos += len(s) + 1
			
			while pos < input_size:
				res += ' '
				pos += 1
				for i in xrange(cLen):
					s = content[i][0]
					if pos + len(s) <= input_size:
						res += s
						pos += len(s)
						if pos + 1 <= input_size:
							res += ' '
							pos += 1
					else:
						idx = 0
						while pos < input_size:
							res += s[idx]
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

def preprocess(input_size):
	
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