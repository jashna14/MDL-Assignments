import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable



def get_data(data , test_set_ratio , train_set_cnt):
	train_data_total,test_data = train_test_split(data,test_size=test_set_ratio)
	size = int(len(train_data_total)/train_set_cnt)
	train_data = []
	for i in range(train_set_cnt - 1):
		train_data_total,train_data_set = train_test_split(train_data_total,test_size=size)
		train_data.append(train_data_set)
	train_data.append(train_data_total)

	data_dic = {}
	data_dic['train_data'] = train_data
	data_dic['test_data'] = test_data

	return data_dic


def get_bias(prediction , actual , model_cnt):
	sum = 0
	for i in range(len(actual)):
		sum += (actual[i] - np.mean(np.array(prediction)[:,i:i+1]))**2
	return sum/len(actual)

def get_variance(prediction, model_cnt):
	sum = 0
	for i in range(len(prediction)):
		sum += np.var(np.array(prediction)[:,i:i+1])
	return sum/len(prediction)

def train_data(data_dic , models_cnt , max_degree ,bias ,variance):

	for i in range(max_degree):
		prediction = []
		poly = PolynomialFeatures(i+1)
		for j in range(models_cnt):
			model = LinearRegression().fit(poly.fit_transform(data_dic['train_data'][j][:,0:1]) , data_dic['train_data'][j][:,1:2])
			pred = model.predict(poly.fit_transform(data_dic['test_data'][:,0:1]))
			prediction.append(pred)
		bias.append(get_bias(prediction , data_dic['test_data'][:,1:2] , models_cnt))
		variance.append(get_variance(prediction, models_cnt))	 




with open('Q1_data/data.pkl', 'rb') as f:
    data = pickle.load(f)

models_cnt =10
max_degree = 9
data_dic = get_data(data ,0.1 , models_cnt)
bias = []
variance = []
train_data(data_dic , models_cnt , max_degree ,bias ,variance)

table = PrettyTable()
table.field_names = ["Degree", "Bias", "Variance"]
for i in range(max_degree):
	table.add_row([ i+1,float(bias[i]),variance[i] ])
print(table)