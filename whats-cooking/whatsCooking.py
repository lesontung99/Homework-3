from numpy import loadtxt, array
import numpy as np
from time import time
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    #Train model
    startTime = time()
    filename = "whats-cooking\dish_train.dat"
    data = loadtxt(filename, delimiter=' ')
    #It tooks 40 seconds just to load all of this lol, I won't test with all of this.
    #Let's extract 2k data first.
    id = array(data[0:5000,0]).T
    y = array(data[0:5000,-1]).T
    x = data[0:5000,0:-1]

   
    print("Data Loading time:", time()-startTime)
    gauss_pipe = Pipeline([
        ('gauss', GaussianNB()),
    ])
    gauss_para = {
        'gauss__priors': [None],
        'gauss__var_smoothing': [1e-10, 1e-9, 5e-10, 5e-9, 1e-8,5e-8],
    }

    bernouli_pipe = Pipeline([
        ('bernouli', BernoulliNB(force_alpha=False,binarize=None)),
    ])
    linear_pipe = Pipeline([
        ('linear', LogisticRegression(dual=False,n_jobs=-1)),
    ])
    bernouli_para = {
        'bernouli__alpha': [0, 0.01 ,0.1 , 1, 10 , 50],
    }
    linear_para = {
        'linear__C':[1,0.5,5],
        

    }
    GaussGrid = GridSearchCV(estimator=gauss_pipe, param_grid=gauss_para,cv=10,verbose=2,n_jobs=-1)
    gaussStart = time()
    GaussGrid.fit(x,y)
    GaussRuntime = time()- gaussStart
    BernouliGrid =  GridSearchCV(estimator=bernouli_pipe, param_grid=bernouli_para,cv=10,verbose=2,n_jobs=-1)
    BernouliStart = time()
    BernouliGrid.fit(x,y)
    BernouliRuntime = time()- BernouliStart
    LinearGrid =  GridSearchCV(estimator=linear_pipe, param_grid=linear_para,cv=10,verbose=2,n_jobs=-1)
    linearStart = time()
    LinearGrid.fit(x,y)
    LinearRuntime =  time()-linearStart
   
    print("GaussianCV accuracy:", GaussGrid.best_score_ )
    print("GaussianCV best parameters:", GaussGrid.best_params_)
    print("Gaussian Runtime: ", GaussRuntime)
    print("BernouliCV accuracy:", BernouliGrid.best_score_)
    print("BernouliCV best parameters:", BernouliGrid.best_params_)
    print("Bernouli Runtime: ", BernouliRuntime)
    #linear_pred = LinearGrid.predict(xtest)
    #linear_acc = accuracy_score(ytest,linear_pred)
    print("Log Regression accuracy:", LinearGrid.best_score_)
    print("Log Regression best parameters:", LinearGrid.best_params_)
    print("Log Regression Runtime", LinearRuntime)



    print("Elapsed time:", time()-startTime)