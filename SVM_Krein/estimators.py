import numpy as np
from utils import quadprog_solve_qp,Krein_EIG
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.gaussian_process.kernels import DotProduct


class SVMK(BaseEstimator,ClassifierMixin):
    def __init__(self,
                kernel = None,
                C = 1):
        self.kernel = kernel
        self.C = C
    def fit(self,X,t):
        labels = np.unique(t).tolist()
        if len(labels)!=2:
            raise ValueError('Estimator for binary classication task.')
        Nc = [sum(t==la) for la in labels]
        whos_min = np.argmin(Nc)
        whos_maj = np.argmax(Nc)
        y = np.ones_like(t)
        y[t==labels[whos_maj]] = -1

        if self.kernel is None:
            self.kernel = DotProduct()
        
        G = self.kernel(X,X)*np.outer(y,y)
        G_t,U,S,D = Krein_EIG(G)
        alpha_t = quadprog_solve_qp(G_t,y,self.C)
        try:
            alpha = (U@S@np.linalg.inv(U + 1e-6)@alpha_t).real    
        except:
            alpha = (U@S@U.T@alpha_t).real
        _SV_index = np.where(alpha != 0)[0]
        self.alpha_SV = alpha[_SV_index]
        self.X_sv = X[_SV_index]
        self.y_sv = y[_SV_index]
        self._SV_index = _SV_index
        
        Ysv = (self.alpha_SV*self.y_sv).reshape(1,-1).repeat(self.X_sv.shape[0],axis=0)
        b = np.mean(self.y_sv - (self.kernel(self.X_sv,self.X_sv)*Ysv).sum(axis=1))
        self.b = b
        return self
