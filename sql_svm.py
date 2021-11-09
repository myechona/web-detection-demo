import os
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from urllib import parse as urlparse
import urllib
import math
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


normal_data = pd.read_csv("pass_.csv", delimiter='\t', encoding='gbk')
abnormal_data = pd.read_csv("injec_.csv", delimiter='\t', encoding='gbk')

def getlen(x):
	return len(x)

def getfirstlen(x):
	parsed_tuple = urlparse.urlparse(urllib.parse.unquote(x))
	url_query = urlparse.parse_qs(parsed_tuple.query,True)
	url_first_arg_len = 0
	if len(url_query) == 0:
		url_first_arg_len = 0
	elif len(url_query) == 1:
		url_first_arg_len = len(url_query[list(url_query.keys())[0]][0])
	else:
		max_len = 0
		for i in url_query.keys():
			if len(url_query[i][0]) > max_len:
				max_len = len(url_query[i][0])
		url_first_arg_len = max_len
	return url_first_arg_len

def getshan(x):
	url = x
	tmp_dict = {}
	url_len = len(url)
	for i in range(0,url_len):
		if url[i] in tmp_dict.keys():
			tmp_dict[url[i]] = tmp_dict[url[i]] + 1
		else:
			tmp_dict[url[i]] = 1
	shannon = 0
	for i in tmp_dict.keys():
		p = float(tmp_dict[i]) / url_len
		shannon = shannon - p * math.log(p,2)
	return shannon

def getchar(x):
	lower = x
	url_ilg_sql = lower.count('select')+lower.count('and')+lower.count('or')+lower.count('insert')+lower.count('update')+lower.count('sleep')+lower.count('benchmark')+\
		lower.count('drop')+lower.count('case')+lower.count('when')+lower.count('like')+lower.count('schema')+lower.count('&&')+lower.count('^')+lower.count('*')+lower.count('--')+lower.count('!')+lower.count('null') +\
		lower.count('%')+lower.count(' ')
	url_ilg_xss = lower.count('script')+lower.count('>')+lower.count('<')+lower.count('&#')+lower.count('chr')+lower.count('fromcharcode')+lower.count(':url')+\
		lower.count('iframe')+lower.count('div')+lower.count('onmousemove')+lower.count('onmouseenter')+lower.count('onmouseover')+lower.count('onload')+lower.count('onclick')+lower.count('onerror')+lower.count('#')+lower.count('expression')+lower.count('eval')
	url_ilg_file = lower.count('./')+lower.count('file_get_contents')+lower.count('file_put_contents')+lower.count('load_file')+lower.count('include')+lower.count('require')+lower.count('open')
	count = url_ilg_sql + url_ilg_xss + url_ilg_file
	return count

def getlabel(x):
	if x == 0:
		return 0
	elif x == 1:
		return 1

normal_data['len'] = normal_data['request'].map(lambda x:getlen(x)).astype(int)
normal_data['first'] = normal_data['request'].map(lambda x:getfirstlen(x)).astype(int)
normal_data['shan'] = normal_data['request'].map(lambda x:getshan(x)).astype(float)
normal_data['char'] = normal_data['request'].map(lambda x:getchar(x)).astype(int)
normal_data['label'] = normal_data['request'].map(lambda x:getlabel(0)).astype(int)
#normal_data = normal_data.drop(['request','len'],axis = 1)
normal_data = normal_data.drop(['request'],axis = 1)

abnormal_data['len'] = abnormal_data['request'].map(lambda x:getlen(x)).astype(int)
abnormal_data['first'] = abnormal_data['request'].map(lambda x:getfirstlen(x)).astype(int)
abnormal_data['shan'] = abnormal_data['request'].map(lambda x:getshan(x)).astype(float)
abnormal_data['char'] = abnormal_data['request'].map(lambda x:getchar(x)).astype(int)
abnormal_data['label'] = abnormal_data['request'].map(lambda x:getlabel(1)).astype(int)
#abnormal_data = abnormal_data.drop(['url','len','id','risk_type','request_time','http_status','http_user_agent','host','cookie_uid','source_ip','destination_ip','last_update_time'],axis = 1)
abnormal_data = abnormal_data.drop(['request'], axis = 1)

train_data = pd.concat([normal_data, abnormal_data], axis = 0)
train_data.info()
train_data.head()
scaler = preprocessing.StandardScaler()
first_scaler_param = scaler.fit(train_data['first'].values.reshape(-1,1))
train_data['first_scaled'] = scaler.fit_transform(train_data['first'].values.reshape(-1,1),first_scaler_param)

lab = train_data['label']
train_data.drop(['label'],axis = 1,inplace = True)
train_data.insert(0,'label',lab)

train_data = shuffle(train_data)
#train = train_data.as_matrix()
train = train_data.iloc[:,:].values

y = train[0:4000,0]
x = train[0:4000,1:]
y_t = train[4000:,0]
x_t = train[4000:,1:]


lr = SVC(kernel='linear',C=0.4).fit(x,y)
res = lr.predict(x_t)

from sklearn.metrics import accuracy_score
print (accuracy_score(res, y_t))
