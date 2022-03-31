# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import warnings
from statsmodels.tsa import stattools as ts
from scipy.optimize import minimize
from ._GPMethods import *


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


def compute_loglike_sum_GP(databins, params):
    loglikesum = []
    for i in range(len(databins)):
        loglikesum.append(compute_loglike_GP(
            databins[i]['X'], databins[i]['y'], params))

    return np.array(loglikesum).sum()


def compute_loglike_grad_sum_GP(databins, params):
    loglikesum = []
    for i in range(len(databins)):
        loglikesum.append(compute_loglike_grad_GP(
            databins[i]['X'], databins[i]['y'], params))

    return np.array(loglikesum).sum(axis=0)


def estimate_binned_params(databins, opt_method='L-BFGS-B'):
    ncov = databins[0]['X'].shape[1]
    theta = [0] * ncov

    for i in range(ncov):
        theta[i] = np.mean([bins['X'][:, i].std(ddof=1) for bins in databins])

    beta = np.mean([bins['y'][:].mean() for bins in databins])
    sigma_f = np.mean([bins['y'][:].std(ddof=1) / math.sqrt(2)
                      for bins in databins])
    sigma_n = np.mean([bins['y'][:].std(ddof=1) / math.sqrt(2)
                      for bins in databins])

    params = {'theta': theta, 'sigma_f': sigma_f,
              'sigma_n': sigma_n, 'beta': beta}

    par_init = []
    par_init.extend(params['theta'])
    par_init.extend([params['sigma_f'], params['sigma_n'], params['beta']])

    obj_fun = lambda par: compute_loglike_sum_GP(databins, params={'theta': par[0:ncov],
                                                                   'sigma_f': par[ncov:ncov + 1], 'sigma_n': par[ncov + 1:ncov + 2], 'beta': par[ncov + 2]})

    obj_grad = lambda par: compute_loglike_grad_sum_GP(databins, params={'theta': par[0:ncov],
                                                                         'sigma_f': par[ncov:ncov + 1], 'sigma_n': par[ncov + 1:ncov + 2], 'beta': par[ncov + 2]})

    optim_result = minimize(fun=obj_fun, x0=par_init,
                            method=opt_method, jac=obj_grad, options={'gtol': 1e-6, 'disp': True})

    estimated_params = {'theta': abs(optim_result.x[0:ncov]),
                        'sigma_f': abs(optim_result.x[ncov:ncov + 1]).item(),
                        'sigma_n': abs(optim_result.x[ncov + 1:ncov + 2]).item(),
                        'beta': optim_result.x[ncov + 2:ncov + 3].item()}

    obj_val = optim_result.fun
    grad_val = optim_result.jac

    return {'estimated_params': estimated_params, 'obj_val': obj_val, 'grad_val': grad_val}


def estimate_parameters(trainX, trainy, optim_size, rng_seed, opt_method='L-BFGS-B', limit_memory=False, optim_idx=None):
    if not limit_memory:
        thinning_number = math.ceil((compute_thinning_number(
            trainX[0]) + compute_thinning_number(trainX[1])) / 2)
        databins1 = create_thinned_bins(trainX[0], trainy[0], thinning_number)
        databins2 = create_thinned_bins(trainX[1], trainy[1], thinning_number)
        databins = np.concatenate([databins1, databins2])
        optim_result = estimate_binned_params(databins, opt_method)
        return optim_result

    elif limit_memory:
        if optim_idx is not None:
            databins = []
            for i in range(len(trainX)):
                databins.append(
                    {'X': trainX[i][optim_idx[i]], 'y': trainy[i][optim_idx[i]]})
        else:
            optim_idx = [None] * len(trainX)
            max_data_sample = optim_size
            tempX = [[]] * len(trainX)
            tempy = [[]] * len(trainX)
            for i in range(len(trainX)):
                if len(trainX[i]) > max_data_sample:
                    np.random.seed(rng_seed)
                    idx = np.random.choice(
                        trainX[i].shape[0], max_data_sample, replace=False)
                    tempX[i] = np.array(trainX[i][idx])
                    tempy[i] = np.array(trainy[i][idx])
                    optim_idx[i] = idx
                else:
                    tempX[i] = trainX[i]
                    tempy[i] = trainy[i]
            databins = []
            for i in range(len(trainX)):
                databins.append({'X': tempX[i], 'y': tempy[i]})

        optim_result = estimate_binned_params(databins, opt_method)
        return optim_result, optim_idx


def compute_diff_cov(trainX, trainy, params, testX, band_size, rng_seed, limit_memory=False, band_idx=None):
    theta = params['theta']
    sigma_f = params['sigma_f']
    sigma_n = params['sigma_n']
    beta = params['beta']

    if limit_memory:
        if band_idx is not None:
            for i in range(len(band_idx)):
                if len(trainX[i]) > band_size:
                    X1, y1 = trainX[i][band_idx[i]], trainy[i][band_idx[i]]
                else:
                    X2, y2 = trainX[i], trainy[i]
        else:
            band_idx = [None] * len(trainX)
            tempX = [[]] * len(trainX)
            tempy = [[]] * len(trainX)
            for i in range(len(trainX)):
                if len(trainX[i]) > band_size:
                    np.random.seed(rng_seed)
                    idx = np.random.choice(
                        trainX[i].shape[0], band_size, replace=False)
                    tempX[i] = trainX[i][idx]
                    tempy[i] = trainy[i][idx]
                    band_idx[i] = idx
                else:
                    tempX[i] = trainX[i]
                    tempy[i] = trainy[i]

            X1, y1 = np.array(tempX[0]), np.array(tempy[0])
            X2, y2 = np.array(tempX[1]), np.array(tempy[1])
    else:
        X1, y1 = trainX[0], trainy[0]
        X2, y2 = trainX[1], trainy[1]

    XT = testX

    return compute_diff_cov_(X1, y1, X2, y2, XT, theta, sigma_f, sigma_n, beta), band_idx


def compute_conf_band(diff_cov_mat, conf_level):
    band = compute_conf_band_(diff_cov_mat, conf_level)
    return np.array(band.tolist())
