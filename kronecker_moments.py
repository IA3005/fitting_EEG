import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
from scipy.linalg.lapack import dtrtri
from scipy.stats import ortho_group, norm, uniform
from scipy.stats import  beta , gengamma,wishart
from scipy.sparse import csr_matrix
from tqdm import tqdm



def generate_random_SPD(p,cond,random_state=None):
    """Generate pxp SPD matrix of the form U^TDU and with conditionning number
    where U is pxp matrix drawn uniformly from the orthogonal group,
    and D is pxp diagonal matrix
    """
    U = ortho_group.rvs(p,random_state=random_state)
    d = np.zeros(p)
    if p>2:
	       d[:p-2] = uniform.rvs(loc=1/np.sqrt(cond),scale=np.sqrt(cond)-1/np.sqrt(cond),size=p-2,random_state=random_state)
    d[p-2] = 1/np.sqrt(cond)
    d[p-1] = np.sqrt(cond)
    return U @ np.diag(d) @ U.T


def commutation_matrix(p1,p2):
    row  = np.arange(p1*p2)
    col  = row.reshape((p1, p2), order='F').ravel()
    data = np.ones(p1*p2, dtype=np.int8)
    com = csr_matrix((data, (row, col)), shape=(p1*p2, p1*p2))
    return com.toarray()

def G(p):
    com = commutation_matrix(p,p)
    return 0.5*((np.kron(com,com)@(np.kron(np.eye(p),np.kron(com,np.eye(p)))))+np.eye(p**4))@(np.kron(com,(np.eye(p**2)+com)@com))

def H(p,k,l):
    return np.kron(np.eye(p**l),np.kron(commutation_matrix(p,p**(k-l-1)),np.eye(p)))

 def J(p,k):
    if k==0:
        return np.zeros((p**(2*k+2),p**(2*k+2)))
    G_ = G(p)
    s = np.zeros((p**(2*(k+1)),p**(2*(k+1))))
    for l in range(k):
        h = H(p,k,l)
        s+= np.kron(h,h)
    A = np.kron(np.eye(p**(k-1)),np.kron(commutation_matrix(p**2,p**(k-1)),np.eye(p**2)))
    B = np.kron(np.eye(p**(k-1)),np.kron(commutation_matrix(p**(k-1),p**2),np.eye(p**2)))
    return s@A@np.kron(np.eye(p**(2*k-2)),G_)@B@np.kron(np.eye(p**k),np.kron(commutation_matrix(p,p**k),np.eye(p)))

def M(p,k):
    B = n*np.kron(np.eye(p**k),np.kron(commutation_matrix(p,p**k),np.eye(p))) + J(p,k)
    return B@commutation_matrix(p**(2*k),p**2)

def kron_power(A,k):
    if k==0:
        return np.eye(1)
    if np.all(A == np.eye(A.shape[0])):
        return np.eye(A.shape[0]**k)
    else:
        if k==1:
            return A
        return np.kron(kron_power(A,k-1),A)

def vectorize(A):
    return A.flatten('F')

def unvectorize(v,p): #transform v into a matrix of shape (p,len(v)//p)
    return v.reshape((len(v)//p, p)).T

    

def kronecker_wishart(p, center_root, k):
    prod = np.eye(p**(2*k))
    for l in range(k):
        A = np.kron(np.eye(p**(2*l)),M(p,k-l-1))
        prod = np.kron(prod,A)
    vec_id = kron_power( vectorize(np.eye(p)), k)
    center_ = kron_power(center_root,2*k)
    return center_ @ prod @ vec_id


#params
n = 5
p = 3
random_state = 123
cond = np.sqrt(10*p)
MC = 10
k = 2

center_root = np.eye(p) #generate_random_SPD(p,cond,random_state=random_state)
center = center_root@center_root


exact_kron = unvectorize(kronecker_wishart(p,center_root,k),p**k)

m2 = n*(np.eye(p**2)+commutation_matrix(p,p))@np.kron(center,center)+n*n*unvectorize(np.kron(vectorize(center),vectorize(center)),p**2)
exact_kron_2 = unvectorize(np.kron(np.eye(p),np.kron(commutation_matrix(p,p),np.eye(p)))@vectorize(m2),p**2)
diffs = []

for j in tqdm(range(MC)):
    samples = wishart.rvs(scale=center,df=n,size=10000,random_state=random_state+j)
    #print(samples.size)
    samples_kron = np.asarray([kron_power(samples[i],k) for i in range(len(samples))])
   # print(samples_kron.shape)
    empirical_kron = np.mean(samples_kron,axis=0)
    #print(np.round(empirical_kron,0))
    diffs.append(np.linalg.norm(exact_kron-empirical_kron))

diffs = np.asarray(diffs)
diffs
        