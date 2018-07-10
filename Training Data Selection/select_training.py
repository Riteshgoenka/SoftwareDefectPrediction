import sys
import numpy as np
from format_data import W
from sklearn import preprocessing
from sklearn.feature_selection import chi2

t_data = []
test_data = []
scores = []
fileList = sys.argv[1:]
num_closest = 2

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
	cols[-1] = 0
	test_data.append(cols)

for fileName in fileList[:-1]:
	tr_data = []
	train_data = []
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
		cols[-1] = 1
		train_data.append(cols)

	data = test_data + train_data
	data = np.array(data)
	x_data = data[:,:-1]
	y_data = data[:,-1]
	min_max_scaler = preprocessing.MinMaxScaler()
	scaled_x_data = min_max_scaler.fit_transform(x_data)
	scores.append(np.linalg.norm(np.multiply(chi2(scaled_x_data, y_data)[0], W)))

for i in np.argpartition(np.array(scores), num_closest)[:num_closest]:
	print(fileList[i])