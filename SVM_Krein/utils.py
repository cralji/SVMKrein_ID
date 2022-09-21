#%% Libraries
import numpy as np
from numpy.random import choice
from numpy.linalg import qr,inv
import quadprog
#%% 
def quadprog_solve_qp(P,y,C,m=None): # , q, G=None, h=None, A=None, b=None):
    if m is None:
        m = P.shape[0]
    dtype = P.dtype
    q = -np.ones((m, ))
    # print('q_dtype: {}'.format(q.dtype))
    G = np.vstack((np.eye(m)*-1,np.eye(m)))
    h = np.hstack((np.zeros(m), np.ones(m) * C))

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    A = y.astype(dtype).reshape(1,-1)
    b = 0.0
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    try:
        alpha = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] 
    except:
        alpha = quadprog.solve_qp(np.eye(m,dtype=np.float64), qp_a, qp_C, qp_b, meq)[0] 
    return alpha

def quadprog_solve_qp_twin(P,C,m=None): # , q, G=None, h=None, A=None, b=None):
    if m is None:
        m = P.shape[0]
    dtype = P.dtype
    q = -np.ones((m, ))
    # print('q_dtype: {}'.format(q.dtype))
    G = np.vstack((np.eye(m)*-1,np.eye(m)))
    h = np.hstack((np.zeros(m), np.ones(m) * C))

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    A = None
    b = None
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    try:
        alpha = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] 
    except:
        alpha = quadprog.solve_qp(np.eye(m,dtype=np.float64), qp_a, qp_C, qp_b, meq)[0] 
    return alpha

def Krein_EIG(K):
    d,U = np.linalg.eig(K)
    D = np.diag(d)
    S = np.sign(D)
    try:
        K_tilde = (U@S@D@np.linalg.inv(U + 1e-6)).real    
    except:
        K_tilde = (U@S@D@U.T).real
    K_tilde = K_tilde.astype(np.float64)
    return K_tilde,U,S,D

#%%
class SimpleSolver:
    """
    SimpleSolver for SVM-based methods
    Implementation based on:
        Vishwanathan, S. V. M., & Narasimha Murty, M. (n.d.). SSVM: a simple SVM algorithm. 
        Proceedings of the 2002 International Joint Conference on Neural Networks. IJCNN’02 (Cat. No.02CH37290). 
        doi:10.1109/ijcnn.2002.1007516 
        https://ieeexplore.ieee.org/abstract/document/1007516
    Implemented by: craljimenez@utp.edu.co

    Input Args:
        G: Quadratic matrix (n x n) from QD Problem from a SVM-based method
        C: regularizer parameter (float)
        y: Default None, it's labels array for the case of SVM-based methods.
        is_SVM: Bool, it specifics if the solver is or not a SVM-based method.
    
    Return Args:
        alpha: array (n,1) Lagrange's Multipliers
    """
    def __init__(self,G,C,y=None,is_SVM=False,tol = 1e-6,max_iter=1000):
        self.G = G
        self.C = C
        self.y = y
        self.is_SVM = is_SVM
        self.tol = tol
        self.max_iter = max_iter
        self.converged = False
        n = self.G.shape[0]
        if self.is_SVM:
            self.Iw = [choice(np.where(self.y == 1)[0],size=1,replace=False)[0],
                choice(np.where(self.y == -1)[0],size=1,replace=False)[0]]
            self.Io = [i for i in range(n) if i not in self.Iw]
            self.IC = []
        else:
            self.Iw = choice(range(n),size=2,replace=False).tolist()
            self.Io = [i for i in range(n) if i not in self.Iw]
            self.IC = []
        if type(C)==int or type(C)==float:
            self.Slacks = C*np.ones((n,))
        else:
            self.Slacks = C
        self.alphas = np.zeros_like(self.Slacks)

    def SolveLinearSystem(self):
        n = self.G.shape[0]
        Gw = self.G[self.Iw]
        nw = len(self.Iw)
        if len(self.IC) == 0:
            GC = self.G[self.IC] # Ic x N 
        else:
            GC = np.zeros((len(self.IC),n))
        Q,R = qr(Gw.dot(Gw.T))
        if self.is_SVM:
            raise ValueError('Not implemented yet')
        else:
            b = Gw.dot(np.ones((n,)) - GC.T.dot(self.Slacks[self.IC]))
        alphas_w = inv(R).dot(Q.T).dot(b) # search inverse from cholesky
        self.alphas[self.Iw] = alphas_w
        return None    

    def simple_solver(self,log = False):
        iter = 0
        error = 1
        self.history={'error':[],
                    'Iw':[],
                    'Io':[],
                    'Iw':[]}
        while iter<self.max_iter and error > self.tol and not(self.converged):
            alphas_old = self.alphas[self.Iw]
            self.SolveLinearSystem()
            alphas = self.alphas[self.Iw]
            #% Activation Constraint
            ind_0 = np.where(alphas<0)[0]
            ind_C = np.where(alphas>self.Slacks[self.Iw])[0]
            tI,ind_min = None,None
            tS,ind_S = None,None
            if (ind_C.size != 0) or (ind_C.size != 0):
                diff = alphas[self.Iw]-alphas_old[self.Iw]
                if ind_0.size != 0:
                    tI,ind_min = np.min(-alphas_old[ind_0]/diff[ind_0]),np.argmin(-alphas_old[ind_0]/diff[ind_0])
                if ind_C.size != 0:
                    tS,ind_S = np.min((self.Slacks[ind_C]-alphas_old[ind_C])/diff[ind_C]),np.argmin((self.Slacks[ind_C]-alphas_old[ind_C])/diff[ind_C])
                
                if (tI is not None) and (tS is None):
                    self.Io.append(self.Iw[ind_0[ind_min]])
                    self.alphas[self.Iw[ind_0[ind_min]]] = 0
                    self.Iw.pop(ind_0[ind_min])

                elif (tS is not None) and (tI is None):
                    self.IC.append(self.Iw[ind_C[ind_S]])
                    self.alphas[self.Iw[ind_C[ind_S]]] = self.Slacks[self.Iw[ind_C[ind_S]]]
                    self.Iw.pop(ind_C[ind_S])
                elif (tI is not None) and (tS is not None):
                    t = np.argmin([tI,tS])
                    if t==0:
                        self.Io.append(self.Iw[ind_0[ind_min]])
                        self.alphas[self.Iw[ind_0[ind_min]]] = 0
                        self.Iw.pop(ind_0[ind_min])
                    else:
                        self.IC.append(self.Iw[ind_C[ind_S]])
                        self.Iw.pop(ind_C[ind_S])
                else:
                    raise ValueError("The is a problem in a BoxConstraints")
                activated = True
            else:
                activated = False
                #% Relaxing Constraints
            if not(activated):
                #% Relaxing Constraing in IC
                Gwc = self.G[self.Iw+self.IC] # (Nw+NC) x n
                if len(self.Io)!=0:
                    G0 = self.G[self.Io] # No x n
                    Aux = G0.dot(Gwc.T.dot(self.alphas[self.Iw+self.IC])-np.ones((len(self.Io),)) )
                    indx = np.where(Aux<0)[0]
                    if indx.size != 0:
                        indx = np.argmin(Aux)
                        self.Iw += [self.Io[indx]]
                        self.Io = [inds for inds in self.Io if inds not in self.Iw]
                #% Relaxing Constraing in IC
                elif len(self.IC) != 0:
                    GC = self.G[self.IC] # Nc x n
                    Aux = GC.dot(Gwc.T.dot(self.alphas[self.Iw+self.C])-np.ones((len(self.IC),)))
                    indx = np.where(Aux>0)[0]
                    if indx.size != 0:
                        indx = np.argmax(Aux)
                        self.Iw += [self.IC[indx]]
                        self.IC = [inds for inds in self.IC if inds not in self.Iw]
                else:
                    self.converged = True
            error = np.linalg.norm(self.alphas-alphas_old)
            iter += 1
            if log:
                print('iter:{} \t error:{} \t nIw:{} \t nIo \t nIC'.format(iter,
                                                                           error,
                                                                           len(self.Iw),
                                                                           len(self.Io),
                                                                           len(self.IC)))
                self.history['error'].append(error)
                self.history['Iw'].append(len(self.Iw))
                self.history['Io'].append(len(self.Io))
                self.history['Iw'].append(len(self.IC))
        return self

#%% Solver AGA https://link.springer.com/article/10.1007/s10957-021-01980-2#Abs1
class solver_AGA:
    def __init__(self,
                 A,
                 a,
                 u,
                 v,
                 epsilon = 1e-2):
        self.A = A
        self.a = a
        self.u = u
        self.v = v
        self.epsilon = epsilon
        self.converged = False
        self.n = A.shape[0]
        self.J = [n for n in range(self.n)]
    
    def assign_x(self):
        x = np.zeros((self.n,))
        if self.is_unconstraint_constant:
            x[self.Jc] = np.random.uniform(self.u,self.v,len(self.Jc))
            x[self.JN] = np.random.choice([self.u,self.v],len(self.JN))
        else:
            for i in self.Jc:
                x[i] = np.random.uniform(self.u[i],self.v[i])
            for i in self.JN:
                x[i] = np.random.choice([self.u[i],self.v[i]])
        return x

    def initializer(self):
        v,_ = np.linalg.eig(self.A)
        if type(v) is complex:
            v = v.real
        mu_1 = np.min(v)
        bar_a = np.diag(self.A) - (np.abs(self.A).sum(axis=1)-np.abs(np.diag(self.A)))
        self.Jc = np.where(bar_a>=0)[0].tolist()
        self.JN = [i for i in range(self.n) if i not in self.Jc]
        if len(self.Jc)==0:
            lambda_ = np.random.uniform(0,1,1)[0]
        else:
            lambda_ = 1
        alpha = lambda_*bar_a + (1-lambda_)*mu_1
        self.bar_alpha = np.minimum(0,alpha)
        self.is_unconstraint_constant = True if ((type(self.u) is float) or (type(self.u) is int)) else False
        self.x = self.assign_x()
    
    def compute_V(self):
        V = np.zeros((self.n,))
        ind1 = np.where(self.x == self.u)[0]
        ind2 = np.where(self.x == self.v)[0]
        ind3 = np.where((self.x > self.u)*(self.x<self.v))[0]
        if self.is_unconstraint_constant:
            V[ind1] = self.E[ind1] + 0.5*self.bar_alpha[ind1]*(self.v - self.u)
            V[ind2] = self.E[ind2] + 0.5*self.bar_alpha[ind2]*(self.u - self.v)
            V[ind3] = self.E[ind3]*self.E[ind3] + 0.5*self.bar_alpha[ind3]*(self.v - self.u)
        else:
            V[ind1] = self.E[ind1] + 0.5*self.bar_alpha[ind1]*(self.v[ind1] - self.u[ind1])
            V[ind2] = self.E[ind2] + 0.5*self.bar_alpha[ind2]*(self.u[ind2] - self.v[ind2])
            V[ind3] = self.E[ind3]*self.E[ind3] + 0.5*self.bar_alpha[ind3]*(self.v[ind3] - self.u[ind3])
        return V

    def compute_betha(self,V):
        ind1 = np.where(V>=0)[0]
        ind2 = np.where(V<0)[0]
        if self.is_unconstraint_constant:
            betha = (V[ind1]*(self.x[ind1]-self.u)).sum() + (V[ind2]*(self.x[ind2]-self.v)).sum()
        else:
            betha = (V[ind1]*(self.x[ind1]-self.u[ind1])).sum() + (V[ind2]*(self.x[ind2]-self.v[ind2])).sum()
        return betha

    def Compute_improved_direction(self):
        ind1 = np.where((self.x==self.u)*(self.V<0)*(self.E<0))[0]
        ind2 = np.where((self.x==self.v)*(self.V>0)*(self.E>0))[0]
        ind3 = np.where((self.x>self.u)*(self.x<self.v)*(self.V!=0)*(self.E<0))[0]
        ind4 = np.where((self.x>self.u)*(self.x<self.v)*(self.V!=0)*(self.E>0))[0]
        l = np.zeros_like(self.V)
        if self.is_unconstraint_constant:
            l[ind1] = self.v - self.u
            l[ind2] = self.u - self.v
            l[ind3] = self.v - self.x[ind3]
            l[ind4] = self.u - self.x[ind4]
        else:
            l[ind1] = self.v[ind1] - self.u[ind1]
            l[ind2] = self.u[ind2] - self.v[ind2]
            l[ind3] = self.v[ind3] - self.x[ind3]
            l[ind4] = self.u[ind4] - self.x[ind4]
        return l

    def Compute_theta_f(self,l):
        lAl = l.T.dot(self.A.dot(l))
        theta_f = np.ones_like(l)
        if lAl > 0:
            ind = np.where(l!=0)[0]
            theta_f[ind] = np.minimum(1,np-abs(self.E*l)/(np.diag(self.A)*l*l))
        return theta_f

    def Compute_direction(self):
        l = np.zeros_like(self.V)
        ind1 = np.where((self.x==self.u)*(self.V<0)*(self.E>=0))[0]
        ind2 = np.where((self.x==self.v)*(self.V>0)*(self.E<=0))[0]
        ind3 = np.where((self.x>self.u)*(self.x<self.v)*(self.V!=0)*(self.E==0))[0]
        if self.is_unconstraint_constant:
            l[ind1] = self.v - self.u
            l[ind2] = self.u - self.v
            l[ind3] = np.minimum(self.v - self.x[ind3], self.u - self.x[ind3])
        else:
            l[ind1] = self.v[ind1] - self.u[ind1]
            l[ind2] = self.u[ind2] - self.v[ind2]
            l[ind3] = np.minimum(self.v[ind3] - self.x[ind3], self.u[ind3] - self.x[ind3])
        return l

    def solver(self):
        self.initializer() # Initilizers algorithms
        k = 0
        while not(self.converged):
            self.E = self.A.dot(self.x) + self.a
            V = self.compute_V()
            self.V = V
            betha = self.compute_betha(V)
            Io = np.where(self.E==0)[0]
            if betha <= self.epsilon:
                self.converged = True
                break
            elif np.all(self.E[Io])<0: # It's not a local minimizer
                l = self.Compute_improved_direction()
                theta_f = self.Compute_theta_f(l)
                self.x += theta_f*l
                k = k+1
            else: # It's a local minimizer
                l = self.Compute_direction()
                theta_f = 1
                self.x += theta_f*l
                k = k+1
        return self.x
                
# %%
A = np.array([[-1,2,0,1],
              [2,-1,1,0],
              [0,1,6,-1],
              [1,0,-1,-2]
             ]
            )
a = np.array([4,9/2,-1,-1])
u = np.array([-1,-1,-1,-1])
v = np.array([1,1,1,1])

aga_solver = solver_AGA(A,a,u,v,epsilon=0.01)
aga_solver.solver()
# %%
