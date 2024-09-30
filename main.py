from moabb.datasets import Kalunga2016
from moabb.paradigms import SSVEP,FilterBankSSVEP
from pyriemann.estimation import Covariances 
import numpy as np
from fitting import fit_eeg
from tqdm import tqdm
import matplotlib.pyplot as plt

tmin = 2 #offset for trials 
delta_t = 3 #considered duration for trials
resampling = 256 #resampling frequency
selected_subjs = [12] #list of selected subjects from the twelve subjects of the dataset
quantity_to_fit = "trace" #change if you want to fit other quantity (see "fitting.py" for more details)
df = {'13':40, '17':35, '21':50, 'rest':23} #degrees of freedom for each class
n_jobs = -1
save_file = True 
plot_cdf = True
path="C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/journal paper/jmva_paper_code-main/fitting results seed=123/"
np.random.seed(123)
    
#1. Load the dataset
dataset = Kalunga2016()
dataset.interval = [tmin,tmin+delta_t]
paradigm = SSVEP()
trials, labels, meta = paradigm.get_data(dataset=dataset,
                                    subjects=selected_subjs)
X = trials[:,:,:-1]
K,p,n = X.shape
    ##X : array of shape (K,p,n)
    ## K=nbr of trials/p=nbr of electrodes/n=nbr of times samples
unique_labels = np.unique(labels)
X = (X- np.tile(X.mean(axis=2).reshape(K,p,1),(1,1,n)))/1e6 #recenter and rescale trials

#2. Extract covariance matrices 
covmats = Covariances().fit_transform(X)*n 
    ##covmats : array of shape (K,p,p), containing K pxp SPDs
cov_data = {}
    ##cov_data : dict associating to each selected subject its covmats and labels
i = 0 
for subject in selected_subjs:
    nb_trials = len(meta[meta['subject']==subject])
    cov_data[subject] = [covmats[i:i+nb_trials],labels[i:i+nb_trials]]
    i += nb_trials


#3. Fitting
xs,eeg_cdf,wishart_cdf,t_wishart_cdf,tests_W , tests_tW = fit_eeg(quantity_to_fit,cov_data,n, df=df,n_jobs=n_jobs)

#4. Plot results of the fitting for each class
if plot_cdf:
    for k in unique_labels:
        plt.plot(xs[k],eeg_cdf[k],label="eeg samples",linestyle="solid")
        plt.plot(xs[k],t_wishart_cdf[k],label="$t$-Wishart samples of df="+str(df[k]),linestyle="dotted")
        plt.plot(xs[k],wishart_cdf[k],label="Wishart samples",linestyle="dashdot")
    
        plt.legend(loc="best",fontsize=7) 
    
        plt.title(f"The empirical CDF of the {quantity_to_fit} for class {k} (df={df[k]})"
                  +f"\n KS p-value of Wishart fitting={tests_W[k][1]}"
                  +f"\n KS p-value of t-Wishart fitting={tests_tW[k][1]}")
    
        plt.show()
      
            
#5. save results in a text file
if save_file:
    for k in unique_labels:
        filename = path+f"cdf_{quantity_to_fit}_class_{k}.txt"
        with open(filename,'w') as f:
           f.write("x,eeg_cdf,wishart_cdf,t_wishart_cdf \n")
           for i in range(len(xs[k])):
               f.write(f"{xs[k][i]},{eeg_cdf[k][i]},{wishart_cdf[k][i]},{t_wishart_cdf[k][i]} \n")
               

