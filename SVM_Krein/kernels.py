#%% Libraries
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.spatial.distance import cdist
from numpy import maximum,sqrt,cos,sin,concatenate
from numpy.random import rand

from utils import GRFF
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
#%% GRFF class

class GRFF_kernel:
    def __init__(self,
                 d=2,
                 s=2,
                 kernel_inizialite = 'gaussian-delta',
                 params_kernel=None):
        self.d = d
        self.s = s
        self.kernel_inizialite = kernel_inizialite
        self.params_kernel = params_kernel
        if kernel_inizialite.lower() == 'gaussian-delta':
            if self.params_kernel is None:
                self.params_kernel = {'a':[-1,1],
                                          'sigmas':[sqrt(d/2)+rand()*1e-4,sqrt(d/2)+rand()*1e-4]
                                         }
        W,kern_pos,kern_neg = GRFF(d=self.d,s=self.s,**self.params_kernel)
        Wpos,Wneg = W[:,:self.s],W[:,self.s:]
        self.Wpos = Wpos
        self.Wneg = Wneg
        self.kern_pos = kern_pos
        self.kern_neg = kern_neg
    
    def __call__(self,X,Y):
        Phi_x = concatenate( (cos(X.dot(self.Wpos)),sin(X.dot(self.Wpos)),-cos(X.dot(self.Wneg)),-sin(X.dot(self.Wneg)) ) )
        Phi_y = concatenate( (cos(Y.dot(self.Wpos)),sin(Y.dot(self.Wpos)),cos(Y.dot(self.Wneg)),sin(Y.dot(self.Wneg)) ) )

        K = Phi_x.dot(Phi_y.T)
        return K