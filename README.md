# Pseudotime Model with Mixture of pPCA

## Intro 
This project explores the modeling of single-cell expression data with a mixture of (Variational Bayesian) probabilistic PCA, allowing for prior information about the sample and time point from which a cell is collected. More details about the model and the current results are in the writeup. 

## Files 
The main scripts for the model is in `mppca_with_priors/`. The implementations of pPCA and mixture of pPCAs are from this [repository](https://github.com/tinrabuzin/MVBPCA), which follows the following papers:
* Bishop, Christopher M. "Variational principal components." (1999): 509-514.
* Bishop, C.M. and Winn, J.M., 2000, June. Non-linear Bayesian image modelling. In European Conference on Computer Vision (pp. 3-17). Springer, Berlin, Heidelberg.

The main difference with the implementation above is the incorporation of cell line and time point prior in `mppca_with_priors/mvbpca.py`. 

The directory `mfa/` contains various implementations onlines of mixture of Factor Analyzers. But I have not had much success making these models work on our data.  