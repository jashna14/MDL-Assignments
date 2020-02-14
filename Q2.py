import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable



def get_data():
    
	with open('Q2_data/X_train.pkl', 'rb') as f:
		x_train = pickle.load(f)

	with open('Q2_data/Y_train.pkl', 'rb') as f:
		y_train = pickle.load(f)

	with open('Q2_data/X_test.pkl', 'rb') as f:
		x_test = pickle.load(f)

	with open('Q2_data/Fx_test.pkl', 'rb') as f:
		y_test = pickle.load(f)

	# x_train[0][1].append(55)
	test = np.array(x_test)
	test_x=test.reshape(-1,1)
	tet_y = np.array(y_test)
	test_y=tet_y.reshape(-1,1)
	# print(test_x,test_y)
	test=np.hstack((test_x,test_y))
	# print(test)
	train=[]



	for i in range(len(x_train)):
		tmp=np.array(x_train[i])
		tmp2=tmp.reshape(-1,1)
		tmp3=np.array(y_train[i])
		tmp4=tmp3.reshape(-1,1)
		tmp5 = np.hstack((tmp2,tmp4))
		train.append(tmp5)



	data_dic = {}
	data_dic['train_data'] = np.array(train)
	data_dic['test_data'] = np.array(test)

	return data_dic


def get_bias(prediction , actual , model_cnt):
	sum = 0
	for i in range(len(actual)):
		sum += (actual[i] - np.mean(np.array(prediction)[:,i:i+1]))**2
	
	return sum/len(actual)


def get_variance(prediction, model_cnt):
	# sum = 0
	# for i in range(len(prediction)):
	# 	sum += np.var(np.array(prediction)[:,i:i+1])
	# return sum/len(prediction[1])

	var=np.var(prediction,axis=0)
	return np.sum(var)/len(prediction[1])


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





models_cnt =20
max_degree = 9
data_dic = get_data()

bias = []
variance = []
train_data(data_dic , models_cnt , max_degree ,bias ,variance)
print(len(data_dic['test_data']))
table = PrettyTable()
table.field_names = ["Degree", "Bias^2", "Variance","Bias"]
degree=[]
total_error=[]
bias_real=np.sqrt(bias)	
for i in range(max_degree):
	table.add_row([ i+1,float(bias[i]),variance[i],float(bias_real[i])])
	total_error.append(variance[i]+bias[i])
	degree.append(i+1)
print(table)
fig, axs = plt.subplots(2,2)
axs[0][0].plot(degree, bias)
axs[0][0].set_ylabel("bias^2")
axs[0][1].plot(degree, variance)
axs[0][1].set_ylabel("variance")
axs[1][0].plot(degree, bias_real)
axs[1][0].set_ylabel("bias")
axs[1][0].set_xlabel("Degree of Polynomial")
axs[1][1].plot(degree,bias,label="bias^2")
axs[1][1].plot(degree,bias_real,label="bias")
axs[1][1].plot(degree,variance,label="variance")
# axs[2].plot(degree,total_error,label="total error")
plt.legend()
axs[1][1].set_ylabel("bias^2-variance")
axs[1][1].set_xlabel("Degree of Polynomial")
plt.show()