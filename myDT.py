"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Le Son Tung:

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from boostedDT import BoostedDT
if __name__ == "__main__":
    # load the data set
    filename = "data\challengeTrainLabeled.dat"
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:10]
    y = np.array([data[:, 10]]).T
    y = y.flatten()
    n,d = X.shape
    nTrain = 0.5*n  #training on 50% of the data

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.5)

    # train the decision tree
    modelDT = BoostedDT(numBoostingIters=150, maxTreeDepth=4)
    modelDT.fit(Xtrain,ytrain)

    # train the boosted DT
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=4)
    modelBoostedDT.fit(Xtrain,ytrain)

    # output predictions on the remaining data
    ypred_DT = modelDT.predict(Xtest)
    ypred_BoostedDT = modelBoostedDT.predict(Xtest)
    # compute the training accuracy of the model
    accuracyDT = accuracy_score(ytest, ypred_DT)
    accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)

    print ("4-Dept Boosted Decision Tree Accuracy = "+str(accuracyDT))
    print ("5-Dept Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT))