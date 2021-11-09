# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
from statsmodels.tsa import stattools as ts


def compute_thinning_number(X):
    ncov = X.shape[1]
    lag_bound = 2/np.sqrt(X.shape[0])
    thinning_vector = np.zeros(ncov)
    for i in range(ncov):
        pacf = ts.pacf(X[:, i], nlags=40)
        thinning_vector[i] = np.argwhere(np.abs(pacf) <= lag_bound).min() + 1
    thinning_number = int(thinning_vector.max())
    return thinning_number


def create_thinned_bins(X, y, thinning_number):
    nrow = X.shape[0]
    thinned_bins = {'X': [], 'y': []}

    if thinning_number < 2:
        thinned_bins['X'].append(X)
        thinned_bins['y'].append(y)

    else:
        for i in range(thinning_number):
            n_points = math.ceil((nrow-i)/thinning_number)
            last_idx = i + (n_points*thinning_number)
            idx = np.arange(i, last_idx, thinning_number)
            thinned_bins['X'].append(X[idx, :])
            thinned_bins['y'].append(y[idx])

    thinned_bins['X'] = np.array(thinned_bins['X'])
    thinned_bins['y'] = np.array(thinned_bins['y'])

    # [thinning_number, total_points/thinning_number, columns]
    return thinned_bins


def estimate_binned_params(databins):
    ncov = databins['X'].shape[2]
    theta = [0]*ncov

    for i in ncov:
        theta[i] = databins['X'][:, :, i].std(axis=1, ddof=1).mean()

    beta = databins['y'].mean()
    sigma_f = (databins['y'].std(axis=1, ddof=1)/math.sqrt(2)).mean()
    sigma_n = (databins['y'].std(axis=1, ddof=1)/math.sqrt(2)).mean()
