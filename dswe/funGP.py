# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_matching
from ._GPMethods import *
from ._funGP_subroutine import *


class FunGP(object):

    """
    Gaussian process aided function comparison using noisy scattered data. 
    Estimates the functions for different data samples using Gaussian process models.

    References
    ----------
    Prakash, A., Tuo, R., & Ding, Y. (2020). "Gaussian process aided 
    function comparison using noisy scattered data." arXiv preprint arXiv:2003.07899. 
    <https://arxiv.org/abs/2003.07899>.

    Parameters
    ----------
    Xlist: list
        A list, consisting of data sets to match, also each of the individual data set can be 
        a matrix with each column corresponding to one input variable.

    ylist: list
        A list, consisting of data sets to match, and each list is an array that corresponds to target 
        values of the data sets.

    testset: np.array
        Test points at which the functions will be compared.

    conf_level: float
        A single value representing the statistical significance level for 
        constructing the band. Default value is 0.95.

    limit_memory: bool
        A boolean (True/False) indicating whether to limit the memory use or not. 
        Default is True. If set to True, 5000 datapoints are randomly sampled 
        from each dataset under comparison for inference.  

    opt_method: string
        A string specifying the optimization method to be used for hyperparameter 
        estimation. The best working solver are ['L-BFGS-B', 'BFGS'].
        Default value is 'L-BFGS-B'.

    sample_size: dict
        A dictionary with two keys: optim_size and band_size, denoting the sample size for each dataset for 
        hyperparameter optimization and confidence band computation, respectively, when limit_memory = TRUE. 
        Default value is {optim_size: 500, band_size: 5000}.

    rng_seed: int
        Random number genrator (rng) seed for sampling data when limit_memory = TRUE. Default value is 1. 

    Returns
    -------
    FunGP
        self with trained parameters. \n
        - mu1: An array of test prediction for first data set.
        - mu2: An array of test prediction for second data set.
        - mu_diff: An array of pointwise difference between the predictions from the two datasets (mu2-mu1).
        - band: An array of the allowed statistical difference between functions at testpoints in testset.
        - conf_level: A numeric representing the statistical significance level for constructing the band.
        - estimated_params: A list of estimated hyperparameters for GP.
    """

    def __init__(self, Xlist, ylist, testset, conf_level=0.95, limit_memory=True, opt_method='L-BFGS-B',
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
                    "If limit_memory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")
            if not set(['optim_size', 'band_size']) == set(list(sample_size.keys())):
                raise ValueError(
                    "If limit_memory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")

        if type(rng_seed) != int:
            raise ValueError(
                "The rng_seed must be a single integer value.")

        if opt_method not in ['L-BFGS-B', 'BFGS']:
            raise ValueError("The opt_method must be 'L-BFGS-B' or 'BFGS'.")

        self.Xlist = Xlist
        self.ylist = ylist
        for i in range(2):
            self.Xlist[i] = np.array(self.Xlist[i])
            self.ylist[i] = np.array(self.ylist[i])

        for i in range(2):
            if len(self.Xlist[i].shape) == 1:
                self.Xlist[i] = self.Xlist[i].reshape(-1, 1)

        if self.Xlist[0].shape[1] != self.Xlist[1].shape[1]:
            raise ValueError(
                "The number of columns in both the dataset should be the same.")

        self.testset = np.array(testset)
        if len(self.testset.shape) == 1:
            self.testset = self.testset.reshape(-1, 1)

        if self.Xlist[0].shape[1] != self.testset.shape[1]:
            raise ValueError(
                "The number of columns in input and testset should be same.")

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
            self.Xlist, self.ylist, self.params, self.testset, self.band_size, self.rng_seed, self.limit_memory)

        self.mu1 = self.diff_cov['mu1']
        self.mu2 = self.diff_cov['mu2']
        self.mu_diff = self.mu2 - self.mu1
        self.band = compute_conf_band(
            self.diff_cov['diff_cov_mat'], self.conf_level)
