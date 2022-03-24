# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_matching
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
    Xlist : A list, consisting of data sets to match, also each of the individual data set can be 
            a matrix with each column corresponding to one input variable.

    ylist : A list, consisting of data sets to match, and each list is a array corresponds to target 
            values of the data sets.

    testX: Test points at which the functions will be compared.

    conf_level: A single value representing the statistical significance level for 
                constructing the band.

    limit_memory: A boolean (True/False) indicating whether to limit the memory use or not. 
                  Default is true. If set to true, 5000 datapoints are randomly sampled 
                  from each dataset under comparison for inference.  

    opt_method: A string specifying the optimization method to be used for hyperparameter 
                estimation. The best working solver are ['L-BFGS-B', 'BFGS'].

    sample_size: A dictionary with two keys: optimSize and bandSize, 
                 denoting the sample size for each dataset for hyperparameter optimization 
                 and confidence band computation, respectively, when limitMemory = TRUE. 
                 Default value is list(optimSize = 500,bandSize = 5000).

    rng_seed: Random seed for sampling data when limitMemory = TRUE. Default is 1. 

    Returns
    -------
    A fitted object (dictionary) of class FunGP.

        mu_diff: An array of pointwise difference between the predictions 
                    from the two datasets (mu2-mu1).
        mu1: An array of test prediction for first data set.
        mu2: An array of test prediction for second data set.
        band: An array of the allowed statistical difference between functions at 
                testpoints in testset.
        conf_level: A numeric representing the statistical significance level for 
                    constructing the band.
        estimated_params: A list of estimated hyperparameters for GP.
    """

    def __init__(self, Xlist, ylist, testX, conf_level=0.95, limit_memory=True, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, rng_seed=1):

        validate_matching(Xlist, ylist)

        if type(conf_level) != int and type(conf_level) != float or conf_level < 0 or conf_level > 1:
            raise ValueError(
                "The conf_level be a numeric value between 0 and 1")

        if type(limit_memory) != type(True):
            raise ValueError("The limit_memory must be either True or False.")

        if limit_memory:
            if not isinstance(sample_size, dict):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")
            if not set(['optim_size', 'band_size']) == set(list(sample_size.keys())):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")

        if type(rng_seed) != int:
            raise ValueError("The range seed must be a single integer value.")

        if opt_method not in ['L-BFGS-B', 'BFGS']:
            raise ValueError("The opt_method must be 'L-BFGS-B' or 'BFGS'.")

        self.Xlist = Xlist
        self.ylist = ylist
        for i in range(2):
            self.Xlist[0] = np.array(self.Xlist[0])
            self.Xlist[1] = np.array(self.Xlist[1])
            self.ylist[0] = np.array(self.ylist[0])
            self.ylist[1] = np.array(self.ylist[1])

        self.testX = np.array(testX)
        self.conf_level = conf_level
        self.limit_memory = limit_memory
        self.opt_method = opt_method
        self.optim_size = sample_size['optim_size']
        self.band_size = sample_size['band_size']
        self.rng_seed = rng_seed

        optim_result = estimate_parameters(self.Xlist, self.ylist, self.optim_size,
                                           self.rng_seed, opt_method=self.opt_method, limit_memory=self.limit_memory)

        self.params = optim_result['estimated_params']

        self.diff_cov = compute_diff_cov(
            self.Xlist, self.ylist, self.params, self.testX, self.band_size, self.rng_seed, self.limit_memory)

        self.mu1 = self.diff_cov['mu1']
        self.mu2 = self.diff_cov['mu2']
        self.mu_diff = self.mu2 - self.mu1
        self.band = compute_conf_band(
            self.diff_cov['diff_cov_mat'], self.conf_level)
