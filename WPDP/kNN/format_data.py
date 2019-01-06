import sys
import numpy as np
from sklearn import preprocessing

data = []
fileList = sys.argv[1:]

for fileName in fileList:
	fil = open(fileName, "r")
	next(fil)
	for line in fil:
		line1 = line.rstrip()
		cols = line1.split(",")
		try:
			cols.remove('\n')
		except:
			pass
		cols = cols[3:]
		for j in range(len(cols)):
			cols[j] = float(cols[j])
		if(cols[-1] > 0):
			cols[-1] = 1
		data.append(cols)

data = np.array(data)
x_data = data[:,:-1]
Y_DATA = data[:,-1]
min_max_scaler = preprocessing.MinMaxScaler()
X_DATA = min_max_scaler.fit_transform(x_data)
n, f = X_DATA.shape