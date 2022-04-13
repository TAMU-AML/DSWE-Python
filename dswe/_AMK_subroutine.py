# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import special
from itertools import combinations


def compute_gaussian_kernel(x, y, lmbda):
    kernel = (1 / np.sqrt(2 * np.pi) * lmbda) * \
        np.exp(-np.power(x - y, 2) / (2 * np.power(lmbda, 2)))
    return kernel


def compute_von_mises_kernel(d, d0, nu):
    kernel = np.exp(nu * np.cos(d - d0)) / (2 * np.pi * special.i0(nu))
    return kernel


def compute_weights(X_train, X_test, bw, n_multi_cov, fixed_cov, cir_cov):
    nrow, ncov = X_train.shape

    if n_multi_cov == 'all':
        weights = np.ones((nrow, 1)) / nrow
        kernel = np.array([1] * nrow)

        for i in range(ncov):
            if i in cir_cov:
                cov_kernel = compute_von_mises_kernel(
                    X_train[:, i], X_test[:, i], bw[i])
                cov_kernel[np.isnan(cov_kernel)] = 0
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel
            else:
                cov_kernel = compute_gaussian_kernel(
                    X_train[:, i], X_test[:, i], bw[i])
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
                    X_train[:, i], X_test[:, i], bw[i])
                cov_kernel[np.isnan(cov_kernel)] = 0
                if cov_kernel.sum() != 0:
                    kernel = kernel * cov_kernel
            else:
                cov_kernel = compute_gaussian_kernel(
                    X_train[i, :], X_test[:, i], bw[i])
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
                        X_train[:, f], X_test[:, f], bw[f])
                    cov_kernel[np.isnan(cov_kernel)] = 0
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel
                else:
                    cov_kernel = compute_gaussian_kernel(
                        X_train[:, f], X_test[:, f], bw[f])
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel

            for j in cov_combination[:, i]:
                if j in cir_cov:
                    cov_kernel = compute_von_mises_kernel(
                        X_train[:, j], X_test[:, j], bw[j])
                    cov_kernel[np.isnan(cov_kernel)] = 0
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel
                else:
                    cov_kernel = compute_gaussian_kernel(
                        X_train[:, j], X_test[:, j], bw[j])
                    if cov_kernel.sum() != 0:
                        kernel = kernel * cov_kernel

            if kernel.sum() != 0:
                weights[:, i] = kernel / kernel.sum()

    return weights
