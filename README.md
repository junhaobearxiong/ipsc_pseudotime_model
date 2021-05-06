# Pseudotime Modeling with Mixture of pPCA

## Intro 
This project explores modeling pseudotime of single-cell expression data with a mixture of (Variational Bayesian) probabilistic PCA, allowing for prior information about the sample and time point from which a cell is collected. More details about the model and the current results are in the [slides](https://docs.google.com/presentation/d/16mtGoHBrI78AF2pWrj1Qvv_s26ecFTH4V7Fd2D4_XX0/edit?usp=sharing) here. 

Currently, the variational updates for the time point prior parameters `theta` and `beta` don't appear to be working (results in non-monotonic ELBO). But when not using any priors, only using the cell line prior, or using but fixing the time point priors, the model appears to largely capture the different cell types, as well as maturaty gradient within them. 

The main limitation of the current model is that the estimated pseudotimes (`x`) for each mixture are independent of each other. Using the time priors make the `x` of each mixture look more coherent, but there is still no underlying constraint that ties together the pseudotime estimates of different mixtures. Future work should mainly be addressing this problem, since it's better to have a notion of pseudotime that's coherent across all cells, not just a subset of the cells (especially since those cells don't constitute a full lineage).

## Files 
The main scripts for the model is in `mppca_with_priors/`. The implementations of pPCA and mixture of pPCAs are based on this [repository](https://github.com/tinrabuzin/MVBPCA), which follows the following papers:
* Bishop, Christopher M. "Variational principal components." (1999): 509-514.
* Bishop, C.M. and Winn, J.M., 2000, June. Non-linear Bayesian image modelling. In European Conference on Computer Vision (pp. 3-17). Springer, Berlin, Heidelberg.

The main difference with the implementation above is the incorporation of cell line and time point prior in `mppca_with_priors/mvbpca.py`. Plots are generated with `mppca_with_priors/plot_celltype_fraction.R`.

The directory `mfa/` contains various implementations (found online) of the Mixture of Factor Analyzers. But I have not had much success making these models work on our data.  