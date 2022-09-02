#%% load libraries
from SVM_Krein.estimators import SVMK
from SVM_Krein.kernels import tanh_kernel,TL1,gaussian_delta

from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import f1_score,balanced_accuracy_score,recall_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import itertools
import tikzplotlib




#%% Load Data

# data = pd.read_excel('./data/Cryotherapy.xlsx').to_numpy()
# X,t = data[:,:-1],data[:,-1]

# labels = np.unique(t)
    
# nC = labels.size
# n_per_class = [sum(t==labels[i]) for i in range(nC)]
# # who_is_minoritary = np.argmin(n_per_class)
# who_is_majority = np.argmax(n_per_class)
# y = np.ones_like(t)
# y[t==labels[who_is_majority]] = -1
# y = y.astype(np.int8)

# del t
# t = y
#%%
from sklearn.datasets import make_moons
IR = 6
root = './imgs_fig/'
for IR in [1,2,3,4,5]:
    noise = 0.1
    X,y  = make_moons(n_samples=400,noise=noise)
    y[y==0] = -1 

    ind_neg = np.where(y==-1)[0]
    ind_pos = np.where(y==1)[0]

    ind_pos = ind_pos[np.random.permutation(ind_pos.size)[0:int(ind_neg.size/IR)]]

    X= np.vstack([X[ind_neg],X[ind_pos]])
    y = np.hstack([y[ind_neg],y[ind_pos]])


    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()


    t = y

    #%%
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    # zscore = StandardScaler()
    # X_z = zscore.fit_transform(X)
    # tsne = TSNE(n_components = 2,init = 'pca',perplexity = 30)
    # X_transform = tsne.fit_transform(X_z)

    # plt.figure(figsize=(8,8))
    # plt.scatter(X_transform[:,0],X_transform[:,1],c=t)
    # plt.show()

    # X = X_transform
    #%% functions 
    def plot_DecisionSpace(model,X,t,e = 1,name = None):
        # X = model[1].transform(XX)
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
            plt.savefig(name+'.pdf')
        plt.show()

    nf = 10
    cv1 = StratifiedKFold(n_splits=nf)
    cv2 = StratifiedKFold(n_splits=nf)
    scores = {'acc': 'accuracy',
            'f1_macro':'f1_macro',
            'f1':'f1'}  

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
        s0 = np.median(pdist(Xtrain))
        kernels = []
        gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]
        for sf in [1,1.5,2,3]:
            for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
                kernels.append( gaussian_delta(params_kernel={'sigmas':s,'a':[1.0,-1.00001]} ) )
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
        name = '{}SVM_krein_f{}_IR-{}'.format(root,i,IR)
        plot_DecisionSpace(gridsearch.best_estimator_,Xtrain,t_train,name = name)


# %%
