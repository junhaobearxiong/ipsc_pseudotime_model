"""
Implementation of Mixture of Factor Analyzers in pymc3, inference performed via ADVI
different from classfic MFA in that each mixture has a specific noise 
"""
import pymc3 as pm
import numpy as np
import pickle

class MixtureFA(object):
    def __init__(self, Y, name, K, trace_iter=1000, advi_iter=10000, output_dir='outputs/'):
        """
        Params
        ------
        Y: (N, G) data matrix, N is the number of cells, G is the number of dimensions
            in the case of raw counts, G is the number of genes
            or could be the number of lower dimensional embedding e.g. PCs
        name: name of this analysis, used to name outputs
        K: number of mixture
        advi_iter: number of iterations for ADVI
        trace_iter: number of samples to generate from posterior distribution
        """
        self.Y = Y
        self.name = name
        self.N, self.G = Y.shape
        self.K = K
        self.trace_iter = trace_iter
        self.advi_iter = advi_iter
        self.output_dir = output_dir


    def fit(self):
        """
        Fit MFA model with ADVI
        MU: (G, K), mean for each gene in each mixture
        TAU: (N, 1), the 1d latent position of each sample
        V: (1, G, K), weights for each mixture
        SIGMA: (1, G, K), sd for each gene in each mixture 
        PI: (K,), mixing coefficients
        """
        with pm.Model() as mfa:
            TAU = pm.Normal('TAU', mu=0, sigma=1, shape=(self.N, 1))
            V = pm.Normal('V', mu=0, sigma=1, shape=(1, self.G, self.K))
            MU = pm.Normal('MU', mu=0, sigma=1, shape=(self.G, self.K))
            PI = pm.Dirichlet('PI', a=np.ones(self.K))
            # TODO: initialize with variance as 1, might need more sensible initializations
            SIGMA = pm.HalfCauchy('SIGMA', beta=5, shape=(1, self.G, self.K), testval=np.ones((1, self.G, self.K)))
            
            # TODO: initialize with list is not recommended by tutorial
            components = [
                pm.Normal.dist(
                               mu=MU[..., k] + pm.math.dot(TAU, V[..., k]),
                               sigma=pm.math.dot(np.ones((self.N, 1)), SIGMA[..., k]),
                               shape=(self.N, self.G)
                )
                for k in range(self.K)
            ]
            lik = pm.Mixture('lik', w=PI, comp_dists=components, observed=self.Y, shape=(self.N, self.G))

            # inference via ADVI
            self.mean_field = pm.fit(method='advi', n=self.advi_iter)

        # with open(self.output_dir + self.name + '_model', 'wb') as f:
        #    pickle.dump(self.mean_field, f)


    def get_posterior(self):
        """
        Get posterior point estimate of each variable by sampling the posterior distribution 
        then taking the mean
        """
        trace = self.mean_field.sample(self.trace_iter)
        self.posterior = {}
        self.posterior['MU'] = trace['MU'].mean(axis=0)
        self.posterior['TAU'] = trace['TAU'].mean(axis=0)
        self.posterior['V'] = trace['V'].mean(axis=0)
        self.posterior['SIGMA'] = trace['SIGMA'].mean(axis=0)
        self.posterior['PI'] = trace['PI'].mean(axis=0)
        self.posterior['R'] = self.get_indicator_posterior()

        with open(self.output_dir + self.name + '_posterior.pkl', 'wb') as f:
            pickle.dump(self.posterior, f)


    def get_indicator_posterior(self):
        """
        Get posterior of the indicator variable \Lambda_n for each sample
        based on the optimized value of all other variables
        This is basically the same as taking the E step in EM for GMM
        except we use the optimized `TAU` (also a latent variable) in computing the mean 

        Returns
        -------
        R: (N, K) the mixture responsibility of each mixture for each sample
        """
        def row_normalize(a):
            return a * (1 / a.sum(axis=1)[:, np.newaxis])
        LIK = [
            pm.Normal.dist(mu=self.posterior['MU'][..., k] + pm.math.dot(self.posterior['TAU'], self.posterior['V'][..., k]), 
                           sigma=pm.math.dot(np.ones((self.N, 1)), self.posterior['SIGMA'][..., k]),
                           shape=(self.N, self.G))
            for k in range(self.K)
        ]
        # (N, K): the likelihood of each sample conditioning on each mixture
        likelihood = np.stack([LIK[k].logp(self.Y).eval().sum(axis=1) for k in range(self.K)], axis=1)
        # (N, K): weigh the likelihood of each mixture by pi
        R = likelihood * self.posterior['PI']
        # (N, K): normalize each row i.e. sample 
        R = row_normalize(R)
        return R