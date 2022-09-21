#%% load libraries
from SVM_Krein.estimators import SVMK
from SVM_Krein.kernels import tanh_kernel,TL1,gaussian_delta

from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import f1_score,balanced_accuracy_score,recall_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import itertools
import tikzplotlib
#%% Load Data
n_samples = 100
sigma = 0.1
x = np.linspace(0,2,n_samples)
t1 = x + sigma*np.random.randn(n_samples)
t2 = 2 - x + sigma*np.random.randn(n_samples)

plt.figure(figsize=(8,8))
plt.plot(x,t1,'+r')
plt.plot(x,t2,'+b')
plt.show()


X = np.vstack([np.hstack([x,x]),np.hstack([t1,t2])]).T
t = np.ones((X.shape[0],))
t[n_samples:] = -1
#%%
plt.figure(figsize=(8,8))
plt.scatter(X[:,0],X[:,1],c=t)
plt.show()
#%% functions 
def plot_DecisionSpace(model,X,t,e = 1,name = None):
    nn = 100
    xmin,xmax = np.max(X[:,0]) + e , np.min(X[:,0]) - e
    ymin,ymax = np.max(X[:,1]) + e , np.min(X[:,1]) - e
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,nn),np.linspace(ymin,ymax,nn))
    XX = np.vstack([xx.flatten(),yy.flatten()]).T
    TT = model.predict(XX)
    tt = TT.reshape(xx.shape)
    plt.contourf(xx,yy,tt)
    plt.scatter(X[:,0],X[:,1],c=t,cmap='bwr')
    if name is not None:
        tikzplotlib.save(name+'.tex')
        plt.savefig(name+'.png')
    plt.show()




# %% TWSVM Krein
nf = 10
cv1 = StratifiedKFold(n_splits=nf)
cv2 = StratifiedKFold(n_splits=nf)

scores = {'acc': 'accuracy',
          'f1_macro':'f1_macro',
          'f1':'f1'}  
i = 1
for train_index, test_index in cv1.split(X, t,t):
    Xtrain,t_train = X[train_index],t[train_index]
    Xtest,t_test = X[test_index],t[test_index]

    C_list = [0.00001,0.0001,0.001,0.1,1,10,100]
    C_ = list(itertools.product(C_list,C_list))
    s0 = np.median(pdist(Xtrain))
    kernels = []
    gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]
    for aa in C_:
        for sf in [1,1.5,2,3]:
            for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
                kernels.append( gaussian_delta(params_kernel={'sigmas':s,'a':[aa[0],aa[1]]} ) )
    # kernels.append( TL1(p = 0.7*X.shape[1]) )
    params_grid = {'C': C_list,
                   'kernel': kernels
                 }

    gridsearch = GridSearchCV(SVMK(),
                              param_grid = params_grid,
                              scoring=scores,
                              refit='f1_macro',
                              cv = cv2,
                              n_jobs=4)
    gridsearch.fit(Xtrain,t_train)

    print('Best Params: {}'.format(gridsearch.best_params_))
    print('Classificacion report\n{}'.format(classification_report(t_test,gridsearch.best_estimator_.predict(Xtest))))
    i += 1
    plot_DecisionSpace(gridsearch.best_estimator_,Xtrain,t_train,name = './imgs/TWSVM_krein_f{}'.format(i))

# %% SVM

from sklearn.svm import SVC
i = 0
for train_index, test_index in cv1.split(X, t,t):
    Xtrain,t_train = X[train_index],t[train_index]
    Xtest,t_test = X[test_index],t[test_index]

    C_list = [0.001,0.1,1,10]
    s0 = np.median(pdist(Xtrain))
    kernels = []
    gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]

    for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
        kernels.append( tanh_kernel(gamma=s[0],coef0=s[1]) )
    
    params_grid = {'C':[0.001,0.1,1,10],
                   'gamma': [2**(s) for s in range(-6,7)]
                  }

    
    gridsearch = GridSearchCV(SVC(),
                              param_grid = params_grid,
                              scoring=scores,
                              refit='f1_macro',
                              cv = cv2,
                              n_jobs=4)
    gridsearch.fit(Xtrain,t_train)

    print('Best Params: {}'.format(gridsearch.best_params_))
    print('Classificacion report\n{}'.format(classification_report(t_test,gridsearch.best_estimator_.predict(Xtest))))
    i += 1
    plot_DecisionSpace(gridsearch.best_estimator_,Xtrain,t_train,name = './imgs/SVM_f{}'.format(i))


