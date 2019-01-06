import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from format_data import X_DATA, Y_DATA, X_TEST_DATA, Y_TEST_DATA, f


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


X_TRAIN = X_DATA
X_TEST = X_TEST_DATA
Y_TRAIN = Y_DATA 
Y_TEST = Y_TEST_DATA

classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_TRAIN, Y_TRAIN) 
Y_PRED = classifier.predict(X_TEST)
CM = confusion_matrix(Y_TEST, Y_PRED)
prec = CM[1][1] / (CM[1][1] + CM[0][1])
rec = CM[1][1] / (CM[1][1] + CM[1][0])
fmes = 2 * prec * rec / (prec + rec)
auc = roc_auc_score(Y_TEST, Y_PRED)
balan = bal(CM)

print('Confusion Matrix')
print(CM)
print('Precision: ' + str(prec))
print('Recall: ' + str(rec))
print('fmeasure: ' + str(fmes))
print('AUC: ' + str(auc))
print('Balance: ' + str(balan))