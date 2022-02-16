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
    X : A matrix with each column corresponding to one input variable. 
        array-like of shape (n_samples, n_features).

    y : A vector with each element corresponding to the output at the corresponding row of X.
        array-like of shape (n_samples,).

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

    range_seed: Random seed for sampling data when limitMemory = TRUE. Default is 1. 

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

    def __new__(cls, X, y, testX, conf_level=0.95, limit_memory=False, opt_method='L-BFGS-B',
                sample_size={'optim_size': 500, 'band_size': 5000}, range_seed=1):

        validate_inputs(X, y)
        validate_features(testX)

        X = np.array(X)
        y = np.array(y)
        testX = np.array(testX)

        optim_size = sample_size['optim_size']
        band_size = sample_size['band_size']

        optim_result = estimate_parameters(
            X, y, optim_size, range_seed, opt_method=opt_method, limit_memory=limit_memory)
        params = optim_result['estimated_params']
        diff_cov = compute_diff_conv(
            X, y, params, testX, band_size, range_seed)
        mu_diff = diff_cov['mu2'] - diff_cov['mu1']
        band = compute_conf_band(diff_cov['diff_cov_mat'], conf_level)

        return {'mu_diff': mu_diff, 'mu2': diff_cov['mu2'], 'mu1': diff_cov['mu1'],
                'band': band, 'conf_level': conf_level, 'estimated_params': params}
