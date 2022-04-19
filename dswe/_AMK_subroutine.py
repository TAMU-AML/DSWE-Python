# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import special
from itertools import combinations
import warnings


def compute_gaussian_kernel(x, y, lmbda):
    kernel = (1 / np.sqrt(2 * np.pi) * lmbda) * \
        np.exp(-np.power(x - y, 2) / (2 * np.power(lmbda, 2)))
    return kernel


def compute_von_mises_kernel(d, d0, nu):
    kernel = np.exp(nu * np.cos(d - d0)) / (2 * np.pi * special.i0(nu))
    return kernel


def compute_weights(X_train, testpoint, bw, n_multi_cov, fixed_cov, cir_cov):
    nrow, ncov = X_train.shape

    if n_multi_cov == 'all':
        weights = np.ones((nrow, 1)) / nrow
        kernel = np.array([1] * nrow)

        for i in range(ncov):
            if i in cir_cov:
                cov_kernel = compute_von_mises_kernel(
                    X_train[:, i], testpoint[i], bw[i])
                cov_kernel[np.isnan(cov_kernel)] = 0
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel
            else:
                cov_kernel = compute_gaussian_kernel(
                    X_train[:, i], testpoint[i], bw[i])
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel

        if kernel.sum() != 0:
            weights[:, 1] = kernel / kernel.sum()

    elif n_multi_cov == 'none':
        weights = np.ones((nrow, ncov)) / nrow
        kernel = np.array([1] * nrow)

        for i in range(ncov):
            if i in cir_cov:
                cov_kernel = compute_von_mises_kernel(
                    X_train[:, i], testpoint[i], bw[i])
                cov_kernel[np.isnan(cov_kernel)] = 0
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel
            else:
                cov_kernel = compute_gaussian_kernel(
                    X_train[i, :], testpoint[i], bw[i])
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel

        if kernel.sum() != 0:
            weights[:, i] = kernel / kernel.sum()

    else:
        non_fixed_cov = list(set(list(range(ncov))) - set(fixed_cov))
        cov_combination = np.array(
            list(combinations(non_fixed_cov, (n_multi_cov - len(fixed_cov)))))
        weights = weights = np.ones((nrow, cov_combination.shape[1])) / nrow

        for i in range(cov_combination.shape[1]):
            kernel = np.array([1] * nrow)
            for f in fixed_cov:
                if f in cir_cov:
                    cov_kernel = compute_von_mises_kernel(
                        X_train[:, f], testpoint[f], bw[f])
                    cov_kernel[np.isnan(cov_kernel)] = 0
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel
                else:
                    cov_kernel = compute_gaussian_kernel(
                        X_train[:, f], testpoint[f], bw[f])
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel

            for j in cov_combination[:, i]:
                if j in cir_cov:
                    cov_kernel = compute_von_mises_kernel(
                        X_train[:, j], testpoint[f], bw[j])
                    cov_kernel[np.isnan(cov_kernel)] = 0
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel
                else:
                    cov_kernel = compute_gaussian_kernel(
                        X_train[:, j], testpoint[f], bw[j])
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel

            if kernel.sum() != 0:
                weights[:, i] = kernel / kernel.sum()

    return weights


# Using nrd0 in place of dpill to get some bandwidth corresponding to each covariate.
def nrd0(x):
    return 0.9 * min(np.std(x, ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349) * len(x)**(-0.2)


def compute_bandwidth(X_train, y_train, cir_cov):
    bandwidth = [0] * X_train.shape[1]
    for i in range(X_train.shape[1]):
        bandwidth[i] = nrd0(X_train[:, i])
    if ~np.isnan(cir_cov).all():
        for circ in cir_cov:
            bandwidth[i] = (bandwidth[1] * np.pi) / 180.
            bandwidth[i] = 1 / (bandwidth[i]**2)

    return bandwidth


def kern_pred(X_train, y_train, X_test, bw, n_multi_cov, fixed_cov, cir_cov):
    if bw == 'dpi':
        bandwidth = compute_bandwidth(X_train, y_train, cir_cov)
        if ~np.isfinite(bandwidth).any():
            for i in np.where(~np.isfinite(bandwidth)):
                bandwidth[i] = np.std(X_train[:, i])
        pred = compute_pred(X_train, y_train, X_test,
                            bandwidth, n_multi_cov, fixed_cov, cir_cov)

    elif bw == 'dpi_gap':
        # will finish this part later.
        return

    else:
        bandwidth = bw
        pred = compute_pred(X_train, y_train, X_test,
                            bandwidth, n_multi_cov, fixed_cov, cir_cov)

    return pred


def compute_pred(X_train, y_train, X_test, bandwidth, n_multi_cov, fixed_cov, cir_cov):
    if ~np.isnan(cir_cov):
        for i in cir_cov:
            X_train[:, i] = (X_train[:, i] * np.pi) / 180.
            X_test[:, i] = (X_test[:, i] * np.pi) / 180.

    pred = [None] * len(X_test)
    for i in range(len(pred)):
        weights = compute_weights(
            X_train, X_test[i, :], bandwidth, n_multi_cov, fixed_cov, cir_cov)
        pred[i] = np.matmul(weights, y_train) / weights.shape[1]

    if ~np.isfinite(pred).any():
        warnings.warn(
            "some of the testpoints resulted in non-finite predictions.")

    return pred
