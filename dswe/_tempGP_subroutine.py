# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import math
import random
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


def estimate_binned_params(databins, fast_computation, optim_control, opt_method='L-BFGS-B'):
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

    obj_fun = lambda par: compute_loglike_sum_tempGP(databins, params={'theta': par[0:ncov],
                                                                       'sigma_f': par[ncov:ncov + 1], 'sigma_n': par[ncov + 1:ncov + 2], 'beta': par[ncov + 2]})

    obj_grad = lambda par: compute_loglike_grad_sum_tempGP(databins, params={'theta': par[0:ncov],
                                                                             'sigma_f': par[ncov:ncov + 1], 'sigma_n': par[ncov + 1:ncov + 2], 'beta': par[ncov + 2]})

    if fast_computation:
        optim_result = adam_optimizer(databins, par_init, optim_control['batch_size'], optim_control['learning_rate'], optim_control['max_iter'],
                                      optim_control['tol'], optim_control['beta1'], optim_control['beta2'], optim_control['epsilon'], optim_control['logfile'])
        estimated_params = {'theta': abs(optim_result[0:ncov]),
                            'sigma_f': abs(optim_result[ncov]),
                            'sigma_n': abs(optim_result[ncov + 1]),
                            'beta': optim_result[ncov + 2]}
        obj_val = None
        grad_val = None

        return {'estimated_params': estimated_params, 'obj_val': obj_val, 'grad_val': grad_val}
    else:
        optim_result = minimize(fun=obj_fun, x0=par_init,
                                method=opt_method, jac=obj_grad)
        estimated_params = {'theta': abs(optim_result.x[0:ncov]),
                            'sigma_f': abs(optim_result.x[ncov:ncov + 1]).item(),
                            'sigma_n': abs(optim_result.x[ncov + 1:ncov + 2]).item(),
                            'beta': optim_result.x[ncov + 2:ncov + 3].item()}
        obj_val = optim_result.fun
        grad_val = optim_result.jac

        return {'estimated_params': estimated_params, 'obj_val': obj_val, 'grad_val': grad_val}


def estimate_local_function_params(trainT, residual):
    theta = np.std(trainT, ddof=1)
    sigma_f = np.std(residual, ddof=1) / np.sqrt(2)
    sigma_n = sigma_f

    par_init = [theta, sigma_f, sigma_n]

    obj_fun = lambda par: compute_loglike_GP(trainT.reshape(-1, 1), residual, params={'theta': par[0:1],
                                                                                      'sigma_f': par[1:2], 'sigma_n': par[2:3], 'beta': 0})

    obj_grad = lambda par: compute_loglike_grad_GP_zero_mean(trainT.reshape(-1, 1), residual, params={'theta': par[0:1],
                                                                                                      'sigma_f': par[1:2], 'sigma_n': par[2:3], 'beta': 0})

    optim_result = minimize(fun=obj_fun, x0=par_init,
                            method='L-BFGS-B', jac=obj_grad)

    estimated_params = {'theta': abs(optim_result.x[0:1]),
                        'sigma_f': abs(optim_result.x[1:2]).item(),
                        'sigma_n': abs(optim_result.x[2:3]).item(),
                        'beta': 0}

    obj_val = optim_result.fun
    grad_val = optim_result.jac

    return {'estimated_params': estimated_params, 'obj_val': obj_val, 'grad_val': grad_val}


def compute_local_function(residual, train_T, test_T, neighbourhood):
    pred = [0] * len(test_T)
    for i in range(len(test_T)):
        distance = abs(test_T[i] - train_T)
        train_idx = np.where(distance < neighbourhood)[0]
        if len(train_idx) > 0:
            if np.var(residual[train_idx]) < np.finfo(np.float64).eps or np.isnan(residual[train_idx]).any():
                msg = "While computing g(t), variance of the training residuals is numerically zero for time index: " + str(
                    test_T[i]) + "\nUsing mean of the response as the prediction."
                warnings.warn(msg)
                pred[i] = np.mean(residual[train_idx])
            else:
                try:
                    params = estimate_local_function_params(
                        train_T[train_idx], residual[train_idx])
                    weighted_res = compute_weighted_y(np.array(train_T[train_idx]).reshape(
                        -1, 1), np.array(residual[train_idx]), params['estimated_params'])
                    pred[i] = predict_GP(np.array(train_T[train_idx]).reshape(-1, 1), weighted_res,
                                         np.array(test_T[i]).reshape(-1, 1), params['estimated_params'])[0]
                except:
                    msg = "While computing g(t), variance of the training residuals is numerically zero for time index: " + str(
                        test_T[i]) + "\nUsing mean of the response as the prediction."
                    warnings.warn(msg)
                    pred[i] = np.mean(residual[train_idx])

        else:
            pred[i] = 0

    return pred


def adam_optimizer(databins, par_init, batch_size, learning_rate, max_iter, tol, beta1, beta2, epsilon, logfile):
    ncov = databins[0]['X'].shape[1]
    params_t = par_init
    par_dict = {'theta': par_init[0:ncov],
                'sigma_f': par_init[ncov],
                'sigma_n': par_init[ncov + 1],
                'beta': par_init[ncov + 2]}
    m_t = np.zeros(len(par_init))
    v_t = np.zeros(len(par_init))

    params_mat = np.zeros((max_iter, len(params_t)))

    t = 0
    while(True):
        sampled_bin = random.randint(0, len(databins) - 1)
        if batch_size < len(databins[sampled_bin]['y']):
            sampled_idx = np.random.choice(
                len(databins[sampled_bin]['y']), batch_size)
        else:
            sampled_idx = list(range(len(databins[sampled_bin]['y'])))
        sample_X = databins[sampled_bin]['X'][sampled_idx]
        sample_y = databins[sampled_bin]['y'][sampled_idx]

        grad_t = compute_loglike_grad_GP(sample_X, sample_y, par_dict)

        m_t = (beta1 * m_t) + ((1 - beta1) * grad_t)
        v_t = (beta2 * v_t) + ((1 - beta2) * (grad_t**2))
        m_hat = m_t / (1 - (beta1**(t + 1)))
        v_hat = v_t / (1 - (beta2**(t + 1)))

        params_prev = params_t
        params_t = params_t - \
            (learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon)))
        params_mat[t, :] = params_t
        t = t + 1

        if np.max(np.abs(params_t - params_prev)) < tol or t == max_iter:
            if logfile:
                pd.DataFrame(params_mat).to_csv(logfile + '.csv', index=False)

            return params_t
