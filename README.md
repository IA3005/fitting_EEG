# Fitting EEG data with Elliptical Wishart distributions

This code reproduces the numerical results about fitting real EEG data with Wishart and t-Wishart distributions

To get the figures and the p-values of the statistical tests provided in the paper, please run "main.py"

The repository contains:
| Name             | Description   |
| -------------    |:-------------:|
| main             | Plots figures of fitting provided in the paper         |
| tWishart         | Draws random samples from the t-Wishart distribution and derives the MLE for the center parameter given a degree of freedom     |  
| manifold         | Framework for Riemannian optimization needed to compute the MLE of t-Wishart samples: manifold of the center parameter   |    
| fitting          | Computes the empirical cumulative density function (cdf) of EEG samples and the cdfs of the fitted Wishart and t-Wishart samples, and yields the Kolmogorov-Smirnov statistical tests for the Wishart and t-Wishart fittings     |   

## Requirements: 
numpy - scipy - matplotlib - moabb - pymanopt - pyriemann - tqdm - joblib
