# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_features, validate_inputs
from ._GPMethods import *
from ._funGP_subroutine import *


class FunGP(object):

    """
    Function comparison using Gaussian Process and Hypothesis testing (FunGP)
    -------------------------------------------------------------------------
    Paper: Gaussian process aided function comparison using noisy scattered data.
    Description: Estimates the functions for different data samples using 
                 Gaussian process models.

    References
    ----------
        Prakash, A., Tuo, R., & Ding, Y. (2020). "Gaussian process aided 
        function comparison using noisy scattered data." arXiv preprint arXiv:2003.07899. 
        <https://arxiv.org/abs/2003.07899>.

    Parameters
    ----------
    conf_level: A single value representing the statistical significance level for 
                constructing the band.
    limit_memory: A boolean (True/False) indicating whether to limit the memory use or not. 
                  Default is true. If set to true, 5000 datapoints are randomly sampled 
                  from each dataset under comparison for inference.            
    opt_method: A string specifying the optimization method to be used for hyperparameter 
                estimation. The best working solver are ['L-BFGS-B', 'BFGS'].
    sample_size: A named list of two integer items: optimSize and bandSize, 
                 denoting the sample size for each dataset for hyperparameter optimization 
                 and confidence band computation, respectively, when limitMemory = TRUE. 
                 Default value is list(optimSize = 500,bandSize = 5000).
    range_seed: Random seed for sampling data when limitMemory = TRUE. Default is 1. 

    """

    def __init__(self, conf_level=0.95, limit_memory=False, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, range_seed=1):

        self.conf_level = conf_level
        self.limit_memory = limit_memory
        self.opt_method = opt_method
        self.optim_size = sample_size['optim_size']
        self.band_size = sample_size['band_size']
        self.range_seed = range_seed

    def fit(self, X, y, testX):
        """Fit the FunGP from the training dataset.

        Parameters
        ----------
        X : A matrix with each column corresponding to one input variable. 
            {array-like, sparse matrix} of shape (n_samples, n_features). 
        y : A vector with each element corresponding to the output at the corresponding
            row of X.
            {array-like, sparse matrix} of shape (n_samples,)
            Target values.
        testX: Test points at which the functions will be compared.

        Returns
        -------
        A fitted object of class FunGP.

        mu_diff: A vector of pointwise difference between the predictions 
                 from the two datasets (mu2-mu1).
        mu1: A vector of test prediction for first data set.
        mu2: A vector of test prediction for second data set.
        band: A vector of the allowed statistical difference between functions at 
              testpoints in testset.
        conf_level: A numeric representing the statistical significance level for 
                    constructing the band.
        estimated_params: A list of estimated hyperparameters for GP.


        """

        self.X = X
        self.y = y
        self.testX = testX

        validate_inputs(self.X, self.y)
        validate_features(self.testX)

        optim_result = estimate_parameters(
            self.X, self.y, self.optim_size, self.range_seed, opt_method=self.opt_method, limit_memory=self.limit_memory)
        self.params = optim_result['estimated_params']
        self.diff_cov = compute_diff_conv(
            self.X, self.y, self.params, self.testX, self.band_size, self.range_seed)
        self.mu_diff = self.diff_cov['mu2'] - self.diff_cov['mu1']
        self.band = compute_conf_band(
            self.diff_cov['diff_cov_mat'], self.conf_level)

        return {'mu_diff': self.mu_diff, 'mu2': self.diff_cov['mu2'], 'mu1': self.diff_cov['mu1'],
                'band': self.band, 'conf_level': self.conf_level, 'estimated_params': self.params}
