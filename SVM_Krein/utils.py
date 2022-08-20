#%%
import numpy as np
import quadprog

import matplotlib.pyplot as plt
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

# GRFF implementation,
# Cristian Jimenez, craljimenez@utp.edu.co
# DOI: 
pi = np.pi
def compute_Normal_pdf(x,d,mean = 0,std = 1):
  pdf = ((1/(std*np.sqrt(2*np.pi)))**d)*np.exp(-0.5*(x-mean)*(x-mean)/(std**2))
  return pdf  

def power_delta_gaussian(ww,d,a=None,s = None,mean= 0):
  if a is None:
    m = 2
    a = [1,-1]
  else:
    m = len(a)
  if s is None:
    s = [np.sqrt(d/2*np.random.rand())+1e-8 for mi in range(m)]
  if len(a)!=len(s):
    raise ValueError('len(a) is different to len(sigmas)')
  
  sum = 0
  for ai,si in zip(a,s):
    std = 1/si
    pdf = compute_Normal_pdf(ww,d,mean = mean,std=std)
    cons =  np.sqrt((2*np.pi)**(d))
    sum += ai*cons*pdf
  return sum

def sampler_weights(cdf,ww,d,s):
  w_un = np.random.uniform(0,1,s)
  W_pos = np.interp(w_un,cdf.reshape(-1,),ww)

  w_pos = np.random.normal(size=(d,s))
  WW_pos = 1/np.sqrt(np.sum(w_pos**2,axis=0))

  WW_pos = np.repeat(WW_pos.reshape(1,-1),d,axis=0)*w_pos

  WW_pos = np.repeat(W_pos.reshape(1,-1),d,axis=0)*WW_pos
  return WW_pos

def GRFF(d,
         s,
         num_points = 10000,
         ww_max = 100,
         sigmas = [1,10],
         a = [1,-1],
         plot_distributions = False):
#   s = int(s/2)
# 
  ww = np.linspace(0,ww_max,num_points)

  # Computer kernel power from P(||w||)
  kernelpower = power_delta_gaussian(ww,d = d,s=sigmas,a = a).reshape(-1,1)
  # kernelpower = (2*pi)**(-d/2)*kernelpower
  # Compute kernel positive and negative ( p+(||w||) and p-(||w||) )
  kernelpower_pos = np.zeros_like(kernelpower)
  kernelpower_neg = np.zeros_like(kernelpower)
  ind_pos = np.where(kernelpower>0)
  ind_neg = np.where(kernelpower<0)

  kernelpower_pos[ind_pos] = kernelpower[ind_pos] #   tensor_scatter_nd_update(kernelpower_pos,ind_pos,gather_nd(kernelpower,ind_pos) )
  kernelpower_neg[ind_neg] = kernelpower[ind_neg] #   tensor_scatter_nd_update(kernelpower_neg,ind_neg,-gather_nd(kernelpower,ind_neg) )

  compute_int = lambda x,y: np.trapz(abs(y.reshape(-1,)),x)
  kernelpower_pos_coeff = compute_int(ww,kernelpower_pos)
  kernelpower_neg_coeff = compute_int(ww,kernelpower_neg)

  kernelpower_pos /= kernelpower_pos_coeff 
  kernelpower_neg /= kernelpower_neg_coeff

  kernelpower_coeff = kernelpower_pos_coeff + kernelpower_neg_coeff

  kernelpower_pos_coeff = kernelpower_pos_coeff/kernelpower_coeff
  kernelpower_neg_coeff = kernelpower_neg_coeff/kernelpower_coeff

  # Compute the cumulative distribution
  pos_cdf = np.cumsum(kernelpower_pos/np.sum(kernelpower_pos))
  neg_cdf = np.cumsum(kernelpower_neg/np.sum(kernelpower_neg))

  pos_cdf,ind_pos_unique = np.unique(pos_cdf.reshape(-1,),return_index=True)
  neg_cdf,ind_neg_unique = np.unique(neg_cdf.reshape(-1,),return_index=True)

  ind_pos_unique = np.unique(ind_pos_unique)
  ind_neg_unique = np.unique(ind_neg_unique)

  ind_pos_unique = np.unravel_index(ind_pos_unique,shape=ww.shape)
  ind_neg_unique = np.unravel_index(ind_neg_unique,shape=ww.shape)

  ww_pos = ww[ind_pos_unique]
  ww_neg = ww[ind_neg_unique]

  # Sampler from p_+ and p_- part  
  W_pos = sampler_weights(pos_cdf,ww_pos,d,s)
  W_neg = sampler_weights(neg_cdf,ww_neg,d,s)
  
  if plot_distributions:
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(ww,kernelpower_pos)
    plt.title('$p_+(||w||)$')
    plt.subplot(1,2,2)
    plt.plot(ww,kernelpower_neg)
    plt.title('$p_-(||w||)$')
    plt.suptitle('Mass density')
    plt.show()

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(ww_pos,pos_cdf)
    plt.title('$F_+(||w||)$')
    plt.subplot(1,2,2)
    plt.plot(ww_neg,neg_cdf)
    plt.title('$F_-(||w||)$')
    plt.suptitle('Cumulate distribution')
    plt.show()
  W = np.concatenate((W_pos,W_neg),axis=1)
  kernelpower_pos_coeff = np.sqrt(kernelpower_pos_coeff)
  kernelpower_neg_coeff = np.sqrt(kernelpower_neg_coeff)
  return W,kernelpower_pos_coeff,kernelpower_neg_coeff

# %%
