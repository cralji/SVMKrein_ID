import numpy as np
from SVM_Krein.utils import quadprog_solve_qp,Krein_EIG,quadprog_solve_qp_twin
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
        self.alphas_SV = alpha[_SV_index]
        self.X_sv = X[_SV_index]
        self.y_sv = y[_SV_index]
        self._SV_index = _SV_index
        
        Ysv = (self.alphas_SV*self.y_sv).reshape(1,-1).repeat(self.X_sv.shape[0],axis=0)
        b = np.mean(self.y_sv - (self.kernel(self.X_sv,self.X_sv)*Ysv).sum(axis=1))
        self.b = b
        self.whos_min = whos_min
        self.whos_maj = whos_maj
        self.labels = labels
        return self
    def predict(self,Xtest):
        Ysv = (self.alphas_SV*self.y_sv).reshape(1,-1).repeat(Xtest.shape[0],axis=0)
        y_est = np.sign(np.sum(self.kernel(Xtest,self.X_sv)*Ysv,axis=1) + self.b)
        t_est = np.ones_like(y_est)*self.labels[self.whos_min]
        t_est[y_est==-1] = self.labels[self.whos_maj]
        return t_est

class TWSVM(BaseEstimator,ClassifierMixin):
    def __init__(self,
                 c1 = 0.001,
                 c2 = 1,
                 kernel = None,
                 scale = 1):
            self.c1 = c1
            self.c2 = c2
            self.kernel = kernel
            self.scale = scale

    def fit(self,X,t):

        labels = np.unique(t).tolist()
        if len(labels)!=2:
            raise ValueError('Estimator for binary classication task.')
        Nc = [sum(t==la) for la in labels]
        whos_min = np.argmin(Nc)
        whos_maj = np.argmax(Nc)
        y = np.ones_like(t)
        y[t==labels[whos_maj]] = -1
        self.labels = [labels[whos_min],labels[whos_maj]]
        indexPos = np.where(y == 1)[0]
        indexNeg = np.where(y == -1)[0]

        X_pos = X[y==1][:]
        X_neg = X[y==-1][:]
        N_pos = X_pos.shape[0]
        N_neg = X_neg.shape[0]
        # print(indexPos.shape,indexNeg.shape)
        sort_index = np.vstack([indexPos.reshape(-1,1),indexNeg.reshape(-1,1)]).reshape(-1,)
        # print(sort_index.shape)
        X = X[sort_index]
        y = y[sort_index]
        Q = X.shape[0]
        # print(X.shape,y.shape)
        if self.kernel is None:
            self.kernel = DotProduct()
        
        u_pos = np.ones((N_pos,1))
        u_neg = np.ones((N_neg,1))

        PHI_pos = self.kernel(X_pos,X) # Npos x N
        PHI_neg = self.kernel(X_neg,X) # Nneg x N

        # print(PHI_pos.shape,u_pos.shape)
        S_pos = np.hstack([PHI_pos,u_pos]).T
        # print(S_pos.shape)
        S_neg = np.hstack([PHI_neg,u_neg]).T

        A_pos = np.linalg.inv(S_pos.dot(S_pos.T) +self.c1*np.eye(Q+1))
        A_neg = np.linalg.inv(S_neg.dot(S_neg.T) +self.c1*np.eye(Q+1))

        Hpos = S_neg.T.dot(A_pos.dot(S_neg))
        # print(Hpos.shape)
        # print(u_pos.shape)
        Hneg = S_pos.T.dot(A_neg.dot(S_pos))
        
        alpha_neg = quadprog_solve_qp_twin(Hpos,self.c2)
        alpha_pos = quadprog_solve_qp_twin(Hneg,self.c2)


        z_pos = -1*A_pos.dot(S_neg.dot(alpha_neg))
        z_neg = -1*A_neg.dot(S_pos.dot(alpha_pos))

        self.wpos = z_pos[:-1]
        self.bpos = z_pos[-1]

        self.wneg = z_neg[:-1]
        self.bneg = z_neg[-1]
        K = self.kernel(X,X)
        self.wpos_norm = np.sqrt(self.wpos.T.dot(K.dot(self.wpos)))
        self.wneg_norm = np.sqrt(self.wneg.T.dot(K.dot(self.wneg)))
        self.X = X

        return self

    def predict(self,X):
        labels = self.labels
        phi = self.kernel(X,self.X)   #rff.transform(X)
        d_pos = np.abs(phi.dot(self.wpos) + self.bpos)/self.wpos_norm
        d_neg = np.abs(phi.dot(self.wneg) + self.bneg)/self.wneg_norm
        print(d_pos.shape,d_neg.shape)
        F = d_neg - d_pos # np.concatenate((d_pos.reshape(-1,1),d_neg.reshape(-1,1)),axis=1)

        # inds_min = np.argmin(F,axis=1)
        # print(inds_min.shape)
        t_est = labels[0]*np.ones((X.shape[0],)) #labels[inds_min]
        t_est[F<=0] =labels[1]
        t_est[F>0] =labels[0]

        return t_est