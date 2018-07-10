import sys
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import chi2

t_data = []
tr_data = []
test_data = []
train_data = []
fileList = sys.argv[1:]

for fileName in fileList[:-1]:
	fil = open(fileName, "r")
	next(fil)
	for line in fil:
		line1 = line.rstrip()
		cols = line1.split(",")
		try:
			cols.remove('\n')
		except:
			pass
		tr_data.append(cols[3:])

for cols in tr_data:
	for j in range(len(cols)):
		cols[j] = float(cols[j])
	if(cols[-1] > 0):
		cols[-1] = 1
	train_data.append(cols)

fileName = fileList[-1]
fil = open(fileName, "r")
next(fil)
count = 0
for line in fil:
		line1 = line.rstrip()
		cols = line1.split(",")
		try:
			cols.remove('\n')
		except:
			pass
		t_data.append(cols[3:])
		count += 1

for cols in t_data:
	for j in range(len(cols)):
		cols[j] = float(cols[j])
	if(cols[-1] > 0):
		cols[-1] = 1
	test_data.append(cols)

data = test_data + train_data
data = np.array(data)
x_data = data[:,:-1]
y_data = data[:,-1]
min_max_scaler = preprocessing.MinMaxScaler()
scaled_x_data = min_max_scaler.fit_transform(x_data)

train_data = scaled_x_data[:-count, :]
test_data = scaled_x_data[-count:, :]
y_train_data = y_data[:-count]
y_test_data = y_data[-count:]
W = chi2(train_data, y_train_data)[0]