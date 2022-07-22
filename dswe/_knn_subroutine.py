# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_best_k(X, y, range_k):
    best_k = None
    max_k = max(range_k)

    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    gcv = [0] * len(range_k)

    for i in range(len(range_k)):
        pred_y = y[indices[:, 0:range_k[i]]].mean(axis=1)
        gcv[i] = np.sqrt(
            np.mean(np.power((y - pred_y) / (1 - (1 / range_k[i])), 2)))

    best_k = range_k[np.argmin(gcv)]
    best_rmse = min(gcv)

    if best_k < max_k:
        return ({'best_k': best_k, 'best_rmse': best_rmse})

    if best_k == max_k:
        range_k = max_k + np.linspace(5, 50, 10, dtype=int)

        return compute_best_k(X, y, range_k)


def _compute_best_subset(X, xcol, y, range_k, best_subset, best_rmse, best_k):
    ncov = len(xcol)
    best_col = None

    for i in range(ncov):
        result = compute_best_k(X[:, best_subset + [xcol[i]]], y, range_k)
        rmse = result['best_rmse']
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = result['best_k']
            best_col = xcol[i]

    return_dict = {'best_subset': best_subset,
                   'best_k': best_k, 'best_rmse': best_rmse}

    if best_col is not None:
        best_subset.append(best_col)
        col_diff = list(set(xcol) - set(best_subset))
        if len(col_diff) > 0:
            return_dict = _compute_best_subset(
                X, col_diff, y, range_k, best_subset, best_rmse, best_k)
        else:
            return_dict = {'best_subset': best_subset,
                           'best_k': best_k, 'best_rmse': best_rmse}

    return return_dict


def compute_best_subset(X, y, range_k):
    best_subset = []
    best_rmse = float("inf")
    best_k = None
    xcol = list(range(X.shape[1]))
    return_dict = _compute_best_subset(
        X, xcol, y, range_k, best_subset, best_rmse, best_k)

    return return_dict
