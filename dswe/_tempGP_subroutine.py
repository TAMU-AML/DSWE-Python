# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
from statsmodels.tsa import stattools as ts
from scipy.optimize import minimize
from ._temp_GP_cpp import *


def compute_thinning_number(X):
    ncov = X.shape[1]
    lag_bound = 2 / np.sqrt(X.shape[0])
    thinning_vector = np.zeros(ncov)
    for i in range(ncov):
        pacf = ts.pacf(X[:, i], nlags=40)
        thinning_vector[i] = np.argwhere(np.abs(pacf) <= lag_bound).min() + 1
    thinning_number = int(thinning_vector.max())
    return thinning_number


def create_thinned_bins(X, y, thinning_number):
    nrow = X.shape[0]
    thinned_bins = []

    if thinning_number < 2:
        thinned_bins.append({'X': X, 'y': y})

    else:
        for i in range(thinning_number):
            n_points = math.ceil((nrow - i) / thinning_number)
            last_idx = i + (n_points * thinning_number)
            idx = np.arange(i, last_idx, thinning_number)
            thinned_bins.append({'X': X[idx, :], 'y': y[idx]})

    return thinned_bins


def compute_loglike_sum_tempGP(databins, params):
    loglikesum = []
    for i in range(len(databins)):
        loglikesum.append(compute_loglike_GP(
            databins[i]['X'], databins[i]['y'], params))

    return np.array(loglikesum).sum()


def compute_loglike_grad_sum_tempGP(databins, params):
    loglikesum = []
    for i in range(len(databins)):
        loglikesum.append(compute_loglike_grad_GP(
            databins[i]['X'], databins[i]['y'], params))

    return np.array(loglikesum).sum(axis=0)


def estimate_binned_params(databins):
    ncov = databins['X'].shape[2]
    theta = [0] * ncov

    for i in ncov:
        theta[i] = databins['X'][:, :, i].std(axis=1, ddof=1).mean()

    beta = databins['y'].mean()
    sigma_f = (databins['y'].std(axis=1, ddof=1) / math.sqrt(2)).mean()
    sigma_n = (databins['y'].std(axis=1, ddof=1) / math.sqrt(2)).mean()
    params = {'theta': theta, 'sigma_f': sigma_f,
              'sigma_n': sigma_n, 'beta': beta}

    par_init = []
    par_init.extend(params['theta'])
    par_init.extend([params['sigma_f'], params['sigma_n'], params['beta']])

    obj_fun = lambda par: compute_loglike_sum_tempGP(databins, params={'theta': par_init[0:ncov],
                                                                       'sigma_f': par_init[ncov:ncov + 1], 'sigma_n': par_init[ncov + 1:ncov + 2], 'beta': par_init[ncov + 2]})

    obj_grad = lambda par: compute_loglike_grad_sum_tempGP(databins, params={'theta': par_init[0:ncov],
                                                                             'sigma_f': par_init[ncov:ncov + 1], 'sigma_n': par_init[ncov + 1:ncov + 2], 'beta': par_init[ncov + 2]})

    optim_result = minimize(fun=obj_fun, x0=par_init,
                            method='L-BFGS-B', jac=obj_grad)

    return optim_result.x
