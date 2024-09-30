import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import numpy as np

from scipy.stats import wishart,beta

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
from scipy.linalg import pinvh

from joblib import Parallel, delayed

from manifold import SPD

import numpy.linalg as la
from scipy.special import digamma,gammaln
from scipy.optimize import root_scalar


### simulate t-Wishart samples

def t_wishart_rvs(n,scale,df,size=1):
    """
    Draw random samples from a t-Wishart distribution.

    Parameters
    ----------
    n : int
        Degrees of freedom, must be greater than or equal to dimension of the scale matrix.
    scale : array_like
        Symmetric positive definite scale matrix of the distribution.
    df : float
        Degrees of freedom of the t- modelling.
    size : int
        Number of samples to draw (defaut 1).

    Returns
    -------
    ndarray
        Random variates of shape (`size`,`dim`, `dim`), where
            `dim` is the dimension of the scale matrix..

    """
    p,_=scale.shape
    assert n>=p,"The degree of freedom `n` must be greater than or equal to dimension of the scale matrix."
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size)
    qs = beta.rvs(a=df/2,b=n*p/2,size=size)
    vec = df*(1/qs-1)/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T) 



### cost and grad for t- Wishart 

def t_wish_cost(R,S,n,df):
    """
    computes the cost function (negative log-likelihood of t-Wishart up to a multiplicative positive constant)

    Parameters
    ----------
    R : array
        Symmetric positive definite matrix, plays the role of the distribution's center.
    S : ndarray
        Samples, must be symmetric definite positive matrices of the same shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    float
        The negative log-likelihood of the samples at `R` (divided by n*number of samples).

    """
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(-(df+n*p)/2*np.log(1+a/df))/n/k


def t_wish_egrad(R,S,n,df):
    """
    Computes the Riemannian gradient of the cost (with respect to the Fisher Information Metric of t-Wishart)    

    Parameters
    ----------
    R : array
        Symmetric positive definite matrix, plays the role of the distribution's center.
    S : ndarray
        Samples, must be symmetric definite positive matrices of the same shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    TYPE
        Riemannian gradient of the cost of samples at `R`.

    """
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',(df+n*p)/(df+a),S)
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)


def t_wish_est(S,n,df):
    """
    computes iteratively the MLE for the center of samples drawn from 
    t-Wishart with parameters n and df using the Riemannian Gradient Decent 
    algorithm as described in [1]

    Parameters
    ----------
    S : ndarry
        samples, symmetric definite matrices.
    n : int
        Degrees of freedom.
    df : int
        Degrees of freedom of the t- modelling.

    Returns
    -------
    array
        MLE of the center parameter.
        
    References
    ----------
    ..[1] I. Ayadi, F. Bouchard and F. Pascal, "Elliptical Wishart Distribution: Maximum Likelihood Estimator from Information Geometry," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096222.

    """
    p = S.shape[1]
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    #
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    init = np.eye(S.shape[-1])
    optimizer = ConjugateGradient(verbosity=0)
    return optimizer.run(problem, initial_point=init).point


def pop_i(samples,n,df,i):
    K =len(samples)
    index_i= list(range(0,i))+list(range(i+1,K))
    cov_i = t_wish_est(samples[index_i],n,df) #simplify
    return pinvh(cov_i)

def pop_parallel(samples,n,maxiter=10,threshold=5e-2):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        inverses_cov = Parallel(n_jobs=-1)(delayed(pop_i)(samples,n,df_old,i) for i in range(K))
        inverses_cov = np.asarray(inverses_cov)
        #assert inverses_cov.shape==(K),p,p),"problem of dimension!"
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    return dfs

def pop(samples,n,maxiter=10,threshold=5e-2,rmt=False):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        inverses_cov =np.zeros((K,p,p))
        for i in range(K):
            index_i= list(range(0,i))+list(range(i+1,K))
            cov_i = t_wish_est(samples[index_i],n,df_old) #simplify
            inverses_cov[i,:,:] = pinvh(cov_i)
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        if rmt:
            theta = (1-p/(n*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    return dfs

def pop_approx(samples,n,maxiter=10,threshold=5e-2,rmt=False):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinvh(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        cov= t_wish_est(samples,n,df_old)
        inverses_cov =np.zeros((K,p,p))
        traces = np.einsum("kij,ji->k",samples,pinvh(cov))
        for i in range(K):
            cov_i = cov - (df_old+n*p)/(df_old+traces[i])*samples[i]/(n*K)#simplify
            inverses_cov[i,:,:] = pinvh(cov_i)
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        if rmt:
            theta = (1-p/(1*K))*theta #correction RMT; to verify theorically
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    return dfs
   
def df_estim_0(samples,n,center):
    eta = np.trace(np.mean(samples,axis=0))/(n*np.trace(center))
    if eta>1:
        return 2*eta/(eta-1)
    else:
        if eta==1:
            return np.inf
    return 1 #if eta<1, this means that df<2 for eg df=1
   
def df_estim_1(samples,n,center):
    #traces of whitened samples
    _,p,_ = samples.shape
    traces = np.einsum("kij,ji->k",samples,pinvh(center))
    kappa = ((n*p)/(n*p+2))*np.mean(traces**2)/(np.mean(traces)**2)-1
    if kappa>0:
        return 4+2/kappa
    else:
        if kappa==0:
            return np.inf
    return 2 #if kappa<0, df<4 for eg we can choose df=2


def derivative_log_lik(df,n,p,traces):
    #derivative of the log-likelihood with respect to df (divided by K the size)
    return -0.5*n*p/df + 0.5*(digamma(0.5*(df+n*p))-digamma(0.5*df))+0.5*(df+n*p)*np.mean(traces/(df+traces))-0.5*np.mean(np.log(1+traces/df))

def log_lik(df,n,p,traces):
    #the log-likelihood with respect to df (divided by K the size)
    return -0.5*n*p*np.log(df) + gammaln(0.5*(df+n*p))-gammaln(0.5*df)+0.5*(df+n*p)*np.mean(np.log(1+traces/df))

    
def df_estim_2(samples,n,center):
    #MLE
    K,p,_ = samples.shape
    traces = np.einsum("kij,ji->k",samples,pinvh(center))
    print(derivative_log_lik(1,n,p,traces))
    print(derivative_log_lik(n*p,n,p,traces))
    return root_scalar(lambda df : derivative_log_lik(df,n,p,traces), bracket=[1, n*p])


"""
center = np.eye(10)
n=50
p=10
samples = t_wishart_rvs(n,center,5,1000)   
traces = np.einsum("kij,ji->k",samples,pinvh(center))
dfs = np.linspace(0.1,10,1000)
ls = [log_lik(df,n,p,traces) for df in dfs]
plt.plot(dfs,ls)
"""

    