import sys
import numpy
from sklearn import preprocessing

data = []
a_data = []
new_data = []
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
		a_data.append(cols[2:])

for x in a_data:
	if x not in data:
		data.append(x)

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
		data.append(cols[2:])
		count += 1

for cols in data:
	cols = cols[1:]
	for j in range(len(cols)):
		cols[j] = float(cols[j])
	
	if(cols[-1] > 0):
		cols[-1] = 1
	new_data.append(cols)

DATA = numpy.array(new_data)
x_data = DATA[:, :-1]
y_data = DATA[:, -1]
min_max_scaler = preprocessing.MinMaxScaler()
scaled_x_data = min_max_scaler.fit_transform(x_data)

X_DATA = scaled_x_data[:-count, :]
X_TEST_DATA = scaled_x_data[-count:, :]
Y_DATA = y_data[:-count]
Y_TEST_DATA = y_data[-count:]
a, f = X_DATA.shape