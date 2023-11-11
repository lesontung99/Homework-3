'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter



class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.max_depth = maxTreeDepth
        self.maxIter = numBoostingIters
        #We declare a list of model. Hell this will cost a bunches of memory but who cares?
        self.model = []
        self.alpha = np.empty(numBoostingIters,dtype = float)
        self.weight = []
        self.classList = []
        
        #TODO
        self.finalModel = None
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        # Step 1: Initialize a weight system:
        n,d = X.shape
        
        classList = np.unique(y)
        self.classList=classList
        class_count = len(classList)
        classWeight = np.ones((n))
        classWeight = classWeight/n
        weightModifier = np.zeros((n))
        # Again, why do everyone said the data takes in class weight
        #Step 2: Boosted train
        
        for iteration_count in range (self.maxIter):
            # Boost the data
            #APply weight:
            
            testModel = DecisionTreeClassifier(max_depth=self.max_depth)

            # Train
            testModel.fit(X,y,sample_weight=classWeight)
            #Test
            self.model.append(testModel)
            y_pred = testModel.predict(X)
            #print(y_pred)
            #Weight manipulation:
            # SInce I used for loops, it will be slow. No way else.
            for i in range (n):
                if y_pred[i] != y[i]:
                    weightModifier[i] = 1
                else:
                    weightModifier[i] = 0
            #print(weightModifier)
            #Apply:
            #Honestly no one ever explain anything to me. What is II operation? 
            #weightModifier = weightModifier/sum(weightModifier)
            err_m = classWeight.dot(weightModifier)/sum(classWeight)
            #print(err_m)
            #Total error. Should decrease. How?
            
            #
            self.alpha[iteration_count] = 0.5*np.log((1-err_m)/err_m) + 0.5*np.log(class_count-1)
            #Reassign weight:
            weightModifier = weightModifier*2 - 1
            classWeight = classWeight*np.exp(self.alpha[iteration_count]*weightModifier)
            
            #Renormalize:
            classWeight = classWeight/sum(classWeight)
            self.weight.append(classWeight)
        # Train complete.       
        #TODO
        self.weight = np.array(self.weight)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        #print(self.weight)
        #print(self.alpha)
        #model = DecisionTreeClassifier(max_depth=self.max_depth)
        n,d = X.shape
        #print(self.alpha)
        #prob_array = np.tile(self.alpha,(self.maxIter,1)).T
        y_pred = []
        for it in range(self.maxIter):
            y1 = self.model[it].predict(X)
            y_pred.append(y1)
        y = np.array(y_pred)

        #print(y)
        #Now we have two array: array of probability and array of result.
        #Let's combine them:
        final = np.empty(n)
        for i in range(n):
            ab = []
            ay = y[:,i]
            for ub in self.classList:
                r = BoostedDT.dirichlet(ay,ub).dot(self.alpha)
                #print(r,end=";")
                
                ab.append(r)  
            #print("")
            maxIndex=ab.index(max(ab))
            final[i]=self.classList[maxIndex]
        return final
    def dirichlet(X,a):
        '''
        Dirichlet function:
        X: Array of N
        a: Argument.
        Return: Array of size N satisfy: a[i] = 1 if x[i] = a and 0 otherwise
        '''
        n = len(X)
        k = np.empty(n)
        for i in range (n):
            if X[i] == a:
                k[i]=1
            else:
                k[i]=0
        return k
