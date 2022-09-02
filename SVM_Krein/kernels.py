#%% Libraries
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.spatial.distance import cdist
from numpy import maximum,exp
#%%
class tanh_kernel:
    def __init__(self,
                gamma=None,
                coef0=1):
        self.gamma = gamma
        self.coef0 = coef0
    
    def __call__(self,X,Y):
        K = sigmoid_kernel(X,Y,gamma = self.gamma,coef0 = self.coef0)
        return K
#%% TL1 kernel https://doi.org/10.1016/j.acha.2016.09.001
class TL1:
    def __init__(self,p=1):
        self.p = p
    def __call__(self, X,Y):
        D = cdist(X,Y,'minkowski',p=1)
        K = maximum(1-D,0)
        return K

#%% Gaussian Delta
class gaussian_delta:
    def __init__(self, params_kernel=None):
        self.params_kernel = params_kernel
    def __call__(self,X,Y):
        if self.params_kernel is None:
            self.params_kernel = {'a':[-1,1],
                                  'sigmas':[X.shape[1]/2, X.shape[1]/2+1e-6]
                                  }
        D = cdist(X,Y,'euclidean')
        K = 0
        for ai,si in zip(self.params_kernel['a'],self.params_kernel['sigmas']):
            K += ai*exp(-0.5*(D*D)/(si**2))
        return K
