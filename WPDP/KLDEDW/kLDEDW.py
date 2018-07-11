import math
import random
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from format_data import X_DATA, Y_DATA, f
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# Constant definitions
b = 2
g = 10
p = 0.1
G = 100
N = 100
maxk = 10
Kfolds = 3
alpha = 0.5
Nfolds = 10
elitism = 0.1
Tourn_size = 5
C = int(N * elitism)
upper_bound_delta = 1


# Function to choose the fittest individual among randomly chosen Tourn_size individuals
def random_choose(l):
	lim = random.sample(l, Tourn_size)
	X = max(lim, key=lambda x: x[1])
	return X[0]


# Function to pick an item from a list with the probability of selection of each element being specified 
def random_pick(l, probabilities):
	x = random.uniform(0, 1)
	cumulative_probability = 0.0
	for item, item_probability in zip(l, probabilities):
		cumulative_probability += item_probability
		if x < cumulative_probability:
			break
	return item


# Function to compute BLX-alpha crossover of two numbers
def blx_alpha(x, y):
	w_max = max(x, y)
	w_min = min(x, y)
	I = w_max - w_min
	res = random.uniform(w_min - I * alpha, w_max + I * alpha)
	if (res < 0):
		res = 0
	return res


# Function to compute BLX-alpha crossover of two matrices
blx_alpha_matrix = np.vectorize(blx_alpha)


# Function to add crossover of individual s and t to the population
def add_crossover(s, t):
	global k_value_Population
	k = int((k_value_Population[s] + k_value_Population[t]) / 2)
	if( k == 0 ):
		k = 1
	k_value_Population.append(k)

	global Weight_Matrix_Population
	Weight_Matrix_Population.append(blx_alpha_matrix(Weight_Matrix_Population[s], Weight_Matrix_Population[t]))


# Function to mutate a weight
def mutate_weight(w):
	delta = random.uniform(0, upper_bound_delta)
	return(random_pick([w, w + delta*w, w - delta*w], [1 - p, p/2, p/2]))


# Function to mutate a weight matrix
mutate_matrix = np.vectorize(mutate_weight)


# Mutation function for nth individual
def mutate(n):
	global Weight_Matrix_Population
	Weight_Matrix_Population[n] = mutate_matrix(Weight_Matrix_Population[n])

	global k_value_Population
	k = k_value_Population[n]
	k = random_pick([k, (k + 1) % maxk], [1 - p, p])
	if( k == 0):
		k = 1
	k_value_Population[n] = k


# Function to compute weighted F-measure
def fmeasure(ConfMat, beta):
	if(ConfMat[1][1] == 0):
		return 0
	precision = ConfMat[1][1] / (ConfMat[1][1] + ConfMat[0][1])
	recall = ConfMat[1][1] / (ConfMat[1][1] + ConfMat[1][0])
	fm = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
	return fm


# Function to compute balance
def bal(cm): 
	re = cm[1][1] / (cm[1][1] + cm[1][0])
	pf = cm[0][1] / (cm[0][1] + cm[0][0])
	balance = 1 - math.sqrt((pf*pf + (1 - re)*(1 - re)) / 2)
	return balance


# Function to compute fitness of nth individual
def fitness(n, x_data, y_data):

	# K fold stratified cross validation
	skf = StratifiedKFold(n_splits=Kfolds)
	cm = np.array([[0, 0], [0, 0]])

	for train_index, test_index in skf.split(x_data, y_data):
		x_train = x_data[train_index]
		x_test = x_data[test_index]
		y_train = y_data[train_index]
		y_test = y_data[test_index]
		y_pred = knn_set(n ,x_train, y_train, x_test)
		cm = np.add(cm, confusion_matrix(y_test, y_pred))

	return (fmeasure(cm, 10))


# Function which returns a list of predicted labels for test data
def knn_set(n ,x_train, y_train, x_test):
	WM = Weight_Matrix_Population[n]
	Kval = k_value_Population[n]
	pred = [knn(x_train, y_train.astype(int), x, WM, Kval) for x in x_test]
	return pred


# Function which returns the predicted label of test point x
def knn(x_train, y_train, x, WM, Kval):
	dist = np.linalg.norm(np.multiply(np.subtract(x_train, x), WM[y_train]), axis=1)
	count = np.sum(y_train[np.argpartition(dist, Kval)[:Kval]])
	if (count + count >= Kval):
		return 1
	else:
		return 0


# N fold stratified cross validation 
SKF = StratifiedKFold(n_splits=Nfolds)
CM = np.array([[0, 0], [0, 0]])

for TRAIN_INDEX, TEST_INDEX in SKF.split(X_DATA, Y_DATA):

	X_TRAIN = X_DATA[TRAIN_INDEX]
	X_TEST = X_DATA[TEST_INDEX]
	Y_TRAIN = Y_DATA[TRAIN_INDEX] 
	Y_TEST = Y_DATA[TEST_INDEX]
	X_RES = X_TRAIN
	Y_RES = Y_TRAIN
	
	# Generating the initial population
	Weight_Matrix_Population = [ np.random.rand(b, f) for n in range(N)]
	k_value_Population = [ (1 + 2 * random.randint(0, 2)) for n in range(N) ]
	upper_bound_delta = 1
	
	# Loop for evolution through G generations
	for i in range(G):

		if ( (i + 1) % g == 0 ):
			upper_bound_delta = upper_bound_delta - g/G

		num_cores = multiprocessing.cpu_count()
		result = Parallel(n_jobs=num_cores)(delayed(fitness)(j, X_RES, Y_RES) for j in range(N))
		
		fitness_value = [[j, result[j]] for j in range(N)]
		fitness_value.sort(key=lambda x: x[1], reverse=True)
		print('Generation ' + str(i) + ': ' + str(fitness_value[0][1]))

		if (i + 1 == G):
			fittest = fitness_value[0][0]
			break

		pop_list = [fi[0] for fi in fitness_value[C:N]]
		fit_copy = fitness_value[:]

		for j in range(N - C):
			s = random_choose(fit_copy)
			t = random_choose(fit_copy)
			add_crossover(s, t)

		for fi in fitness_value[1:C]:
			mutate(fi[0])

		for j in sorted(pop_list, reverse=True): 
			del Weight_Matrix_Population[j]
			del k_value_Population[j]

		for j in range(C, N):
			mutate(j)

	print('*************************************')

	Y_PRED = knn_set(fittest, X_RES, Y_RES, X_TEST)
	CM = np.add(CM, confusion_matrix(Y_TEST, Y_PRED))

prec = CM[1][1] / (CM[1][1] + CM[0][1])
rec = CM[1][1] / (CM[1][1] + CM[1][0])
fmes = 2 * prec * rec / (prec + rec)
balan = bal(CM)
print('Confusion Matrix')
print(CM)
print('Precision: ' + str(prec))
print('Recall: ' + str(rec))
print('fmeasure: ' + str(fmes))
print('Balance: ' + str(balan))