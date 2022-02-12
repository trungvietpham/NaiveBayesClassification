import numpy as np
from sklearn.model_selection import train_test_split 
import sys, os
path = os.path.join(os.path.dirname(__file__))
sys.path.insert(1, path)
from NB import GaussianNaiveBayes, MultinomialNaiveBayes
folder = "data/"
X = np.load(folder+"X.npy")
y = np.load(folder+"y.npy")
X_data, X_test, y_data, y_test = train_test_split(X, y, test_size = 0.1)

print("Train set: ")
print(X_data.shape)
print (y_data.shape[0])
print("Test set: \n"+str(X_test.shape)+"\n"+str(y_test.shape[0]))
GNB = GaussianNaiveBayes(X_data, y_data)
GNB.fit(X_data,y_data)
y_pred_GNB = GNB.predict(X_test)
print("GaussianNB:  Accuracy: ")
accuraccy = sum(y_pred_GNB==y_test)/y_test.shape[0]
print(str(round(accuraccy*100, 2))+"%")
MNB = MultinomialNaiveBayes(X_data, y_data)
MNB.fit(X_data, y_data)
Xtest = np.array(X_test[0])
y_pred_MNB = MNB.predict(X_test)
print("MultinomialNB:  Accuracy: ")
accuraccy = sum(y_pred_MNB==y_test)/y_test.shape[0]
print(str(round(accuraccy*100, 2))+"%")