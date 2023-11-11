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
    Xtrain = X[idx]
    ytrain = y[idx]

    # split the data
    filename = "data\challengeTestUnlabeled.dat"
    data = np.loadtxt(filename,delimiter=',')
    Xtest = data[:, 0:10]
    # train the decision tree
    # train the boosted DT
    
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
    modelBoostedDT.fit(Xtrain,ytrain)

    #There is 6000 entry on xtest.
    n1,d = Xtest.shape
    n2,d = Xtrain.shape
    repeatCount = (n1+n2-1)//n2
    for trial in range (repeatCount):
        ypred_BoostedDT = modelBoostedDT.predict(Xtest[trial:(max(trial+n2,n1)),:])
        for i in range(len(ypred_BoostedDT)):
            print(int(ypred_BoostedDT[i]),end=",",sep="")
    # output predictions on the remaining data
    
    
    
    # compute the training accuracy of the model
    