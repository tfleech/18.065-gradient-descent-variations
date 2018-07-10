import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import time
import copy

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

num_features = 2
num_labels = 1

train_data = []#np.array([[0,1],[2,3],[5,6],[10,25]])
train_labels = []#np.array([0,0,1,1])

test_data = []#np.array([[11,2],[1,25],[4,5],[6,10],[-1,2]])
test_labels = []#np.array([1,0,1,1,0])

positive_points_filename = 'class1.csv'#'CHF_data.csv'
negative_points_filename = 'class2.csv'#'COPD_data.csv'

#Read in data from csv files
with open(positive_points_filename, 'r') as f:
  reader = csv.reader(f)
  train_data.extend([[float(x[0]),float(x[1])] for x in list(reader)])
train_labels.extend([1 for i in range(len(train_data))])

with open(negative_points_filename, 'r') as f:
	reader = csv.reader(f)
	train_data.extend([[float(x[0]),float(x[1])] for x in list(reader)])
train_labels.extend([0 for i in range(len(train_data) - len(train_labels))])

#build test and train datasets
test_indexes = np.arange(0,len(train_data), 10)
for index in sorted(test_indexes, reverse=True):
	test_data.append(train_data[index])
	del train_data[index]
	test_labels.append(train_labels[index])
	del train_labels[index]

W = [0,0]
b = 0

#The following is a variety of training functions, each implementing a variation of gradient descent
def SGD(train_data, train_labels, W, b, n=0.05, time=True):
	errors = []
	batch_indexes = np.random.choice(len(train_data), len(train_data), replace=False)
	counter = 0
	for point in batch_indexes:
		y_ = train_labels[point]

		x = train_data[point][0]
		y = train_data[point][1]
		e = np.exp(-W[0]*x-W[1]*y-b)
		W[0] = W[0]-n*2*x*e*(1/(e+1) - y_)/(e+1)**2
		W[1] = W[1]-n*2*y*e*(1/(e+1) - y_)/(e+1)**2
		b = b-n*2*e*(1/(e+1) - y_)/(e+1)**2

		if not time:
			if (counter%500==0):
				errors.append(error(W,b,train_data,train_labels))
			counter+=1
	return (W, b, errors)

def Mini_Batch_SGD_with_repl_with_avg(train_data, train_labels, W, b, batch_size, n=0.05, time=True):
	W0_error = 0
	W1_error = 0
	b_error = 0

	batch_indexes = np.random.choice(len(train_data), batch_size, replace=False)
	counter = 0
	for point in batch_indexes:
		y_ = train_labels[point]

		x = train_data[point][0]
		y = train_data[point][1]
		e = np.exp(-W[0]*x-W[1]*y-b)
		W0_error += 2*x*e*(1/(e+1) - y_)/(e+1)**2
		W1_error += 2*y*e*(1/(e+1) - y_)/(e+1)**2
		b_error += 2*e*(1/(e+1) - y_)/(e+1)**2
	W[0] -= n*(W0_error/batch_size)
	W[1] -= n*(W1_error/batch_size)
	b -= n*(b_error/batch_size)

	return (W, b)

def Mini_Batch_SGD_with_repl_without_avg(train_data, train_labels, W, b, batch_size, n=0.05, time=True):
	batch_indexes = np.random.choice(len(train_data), batch_size, replace=False)
	counter = 0
	for point in batch_indexes:
		y_ = train_labels[point]

		x = train_data[point][0]
		y = train_data[point][1]
		e = np.exp(-W[0]*x-W[1]*y-b)
		W[0] -= n*2*x*e*(1/(e+1) - y_)/(e+1)**2
		W[1] -= n*2*y*e*(1/(e+1) - y_)/(e+1)**2
		b -= n*2*e*(1/(e+1) - y_)/(e+1)**2

	return (W, b)

def Mini_Batch_SGD_without_repl_without_avg(train_data, train_labels, W, b, batch_size, n=0.05, time=True):

	if batch_size > len(train_data):
		batch_size = len(train_data)
	batch_indexes = np.random.choice(len(train_data), batch_size, replace=False)

	for point in batch_indexes:
		y_ = train_labels[point]

		x = train_data[point][0]
		y = train_data[point][1]
		e = np.exp(-W[0]*x-W[1]*y-b)
		W[0] -= n*2*x*e*(1/(e+1) - y_)/(e+1)**2
		W[1] -= n*2*y*e*(1/(e+1) - y_)/(e+1)**2
		b -= n*2*e*(1/(e+1) - y_)/(e+1)**2

	for point in sorted(batch_indexes, reverse=True):
		del train_data[point]
		del train_labels[point]

	return (W, b, train_data, train_labels)

def Mini_Batch_SGD_without_repl_with_avg(train_data, train_labels, W, b, batch_size, n=0.05, time=True):
	W0_error = 0
	W1_error = 0
	b_error = 0

	if batch_size > len(train_data):
		batch_size = len(train_data)
	batch_indexes = np.random.choice(len(train_data), batch_size, replace=False)

	for point in batch_indexes:
		y_ = train_labels[point]

		x = train_data[point][0]
		y = train_data[point][1]
		e = np.exp(-W[0]*x-W[1]*y-b)
		W0_error += 2*x*e*(1/(e+1) - y_)/(e+1)**2
		W1_error += 2*y*e*(1/(e+1) - y_)/(e+1)**2
		b_error += 2*e*(1/(e+1) - y_)/(e+1)**2

	W[0] -= n*(W0_error/batch_size)
	W[1] -= n*(W1_error/batch_size)
	b -= n*(b_error/batch_size)

	for point in sorted(batch_indexes, reverse=True):
		del train_data[point]
		del train_labels[point]

	return (W, b, train_data, train_labels)

#Helper functions to evaluate model
def evaluate_model(W,b,point):
	x = point[0]
	y = point[1]
	z = sigmoid(W[0]*x + W[1]*y + b)
	return z>0.5
def error(W,b,test_data, test_labels):
	accum_error = 0
	for i in range(len(test_data)):
		z = evaluate_model(W,b,test_data[i])
		accum_error += (z==test_labels[i])
	return 1-accum_error/len(test_data)

def print_labels(W,b,data):
	for i in range(len(data)):
		print("point: " + str(data[i]) + " - label: " + str(evaluate_model(W,b,data[i])))

# def train_with_batch_replacement(W,b,train_data,train_labels,batch_size,epochs):
# 	errors = []
# 	for i in range(epochs):
# 		batch_data = []
# 		batch_labels = []
# 		batch_indexes = np.random.choice(len(train_data), batch_size, replace=True)
# 		for index in batch_indexes:
# 			batch_data.append(train_data[index])
# 			batch_labels.append(train_labels[index])
# 		W,b,e = SGD(batch_data, batch_labels, W, b)
# 		errors.extend(e)
# 	return(W,b,errors)


#Functions to run each of the training algorithms
def train_SGD(W,b,train_data,train_labels,epochs,sample_rate=1,time=True):
	errors = []
	for i in range(epochs):
		W,b,e = SGD(train_data, train_labels, W, b, time=time)

		errors.extend(e)
	return(W,b,errors)

def train_mini_batch_with_repl_with_avg(W,b,train_data,train_labels,epochs,sample_rate=1,time=True, batch_size=100):
	errors = []
	for i in range(epochs):
		W,b = Mini_Batch_SGD_with_repl_with_avg(train_data, train_labels, W, b, batch_size)
		if not time:
			errors.append(error(W,b,train_data, train_labels))
	return(W,b,errors)

def train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,epochs,sample_rate=1,time=True, batch_size=100):
	errors = []
	for i in range(epochs):
		W,b = Mini_Batch_SGD_with_repl_without_avg(train_data, train_labels, W, b, batch_size)
		if not time:
			errors.append(error(W,b,train_data, train_labels))
	return(W,b,errors)

def train_mini_batch_without_repl_without_avg(W,b,train_data,train_labels,epochs,sample_rate=1,time=True):
	errors = []
	working_train_data = copy.deepcopy(train_data)
	working_train_labels = copy.deepcopy(train_labels)
	for i in range(epochs):
		W,b,working_train_data,working_train_labels = Mini_Batch_SGD_without_repl_without_avg(working_train_data, working_train_labels, W, b, 100)
		if not time:
			errors.append(error(W,b,train_data, train_labels))
	return(W,b,errors)

def train_mini_batch_without_repl_with_avg(W,b,train_data,train_labels,epochs,sample_rate=1,time=True):
	errors = []
	working_train_data = copy.deepcopy(train_data)
	working_train_labels = copy.deepcopy(train_labels)
	for i in range(epochs):
		W,b,working_train_data,working_train_labels = Mini_Batch_SGD_without_repl_with_avg(working_train_data, working_train_labels, W, b, 100)
		if not time:
			errors.append(error(W,b,train_data, train_labels))
	return(W,b,errors)

#for i in range(100):
#	W,b = SGD(train_data, train_labels, W, b)


#Code to time different functions
#start = time.time()
#W,b,a = train_SGD(W,b,train_data,train_labels,1,time=False)
#W,b,a = train_mini_batch_with_repl_with_avg(W,b,train_data,train_labels,180,time=False)
#end = time.time()


# print("elapsed time: " + str(end-start))
# print(error(W,b,test_data, test_labels))
# print(W,b)
# x = range(0, len(train_data), 100)
# plt.plot(x,a)
# plt.show()

#Code to generate plots for the convergence based on batch_size
W1,b1,e1 = train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,1800,sample_rate=1,time=False, batch_size=10)
W2,b2,e2 = train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,180,sample_rate=1,time=False, batch_size=100)
W3,b3,e3 = train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,36,sample_rate=1,time=False, batch_size=500)
W4,b4,e4 = train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,18,sample_rate=1,time=False, batch_size=1000)
W5,b5,e5 = train_mini_batch_with_repl_without_avg(W,b,train_data,train_labels,9,sample_rate=1,time=False, batch_size=2000)

x1 = range(0, len(train_data), 10)
x2 = range(0, len(train_data), 100)
x3 = range(0, len(train_data), 500)
x4 = range(0, len(train_data), 1000)
x5 = range(0, len(train_data), 2000)

fig, ax =  plt.subplots()
ax.plot(x1,e1,'g',label='batch size = 10')
ax.plot(x2,e2,'r',label='batch size = 100')
ax.plot(x3,e3,'b',label='batch size = 500')
ax.plot(x4,e4,'c',label='batch size = 1000')
ax.plot(x5,e5,'m',label='batch size = 2000')

legend = ax.legend()

plt.show()

print(error(W1,b1,test_data, test_labels))
print(error(W2,b2,test_data, test_labels))
print(error(W3,b3,test_data, test_labels))
print(error(W4,b4,test_data, test_labels))
print(error(W5,b5,test_data, test_labels))
