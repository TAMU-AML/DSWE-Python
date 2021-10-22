# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

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
            np.mean(np.power((y - pred_y) / (1 - (1/range_k[i])), 2)))

    best_k = range_k[np.argmax(gcv)]
    best_rmse = min(gcv)

    if best_k < max_k:
        return {'best_k': best_k, 'best_rmse': best_rmse}

    if best_k == max_k:
        range_k = max_k + np.linspace(5, 50, 10, dtype=int)

        return compute_best_k(X, y, range_k)


def compute_best_subset(X, y, range_k):
    best_subset = None
    best_rmse = float("inf")
    best_k = None

    def _compute_best_subset(X, y, range_k, best_subset, best_rmse, best_k):
        # cols = X.shape[1]
        pass

    pass
