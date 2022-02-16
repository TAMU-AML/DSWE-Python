# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import warnings
from scipy import stats


def generate_test_set(data, grid_size):
    if data[0].shape[1] == 1:
        var1min = max([np.quantile(x[:, 0], 0.025) for x in data])
        var1max = min([np.quantile(x[:, 0], 0.975) for x in data])
        var1range = np.linspace(var1min, var1max, grid_size)
        return var1range

    elif data[0].shape[1] == 2:
        var1min = max([np.quantile(x[:, 0], 0.025) for x in data])
        var1max = min([np.quantile(x[:, 0], 0.975) for x in data])
        var1range = np.linspace(var1min, var1max, grid_size[0])
        var2min = max([np.quantile(x[:, 1], 0.025) for x in data])
        var2max = min([np.quantile(x[:, 1], 0.975) for x in data])
        var2range = np.linspace(var2min, var2max, grid_size[1])
        return np.c_[var1range, var2range]
    else:
        warnings.warn("Maximum of two feature columns to be used.")


def compute_diff(mu1, mu2, baseline):
    if baseline == 1:
        avg_mu = np.mean(mu1)
    elif baseline == 2:
        avg_mu = np.mean(mu1)
    else:
        avg_mu = (np.mean(mu1) + np.mean(mu2)) / 2

    diff = np.mean(mu2 - mu1)
    percent_diff = round((diff / avg_mu) * 100, 2)

    return percent_diff


def compute_stat_diff(mu1, mu2, band, baseline):
    if baseline == 1:
        avg_mu = np.mean(mu1)
    elif baseline == 2:
        avg_mu = np.mean(mu1)
    else:
        avg_mu = (np.mean(mu1) + np.mean(mu2)) / 2

    diff_mu = mu2 - mu1

    if (np.abs(diff_mu) < band).any():
        print(diff_mu[np.abs(diff_mu) < band])
        diff_mu[np.abs(diff_mu) < band] = 0
    if (diff_mu > 0).any():
        diff_mu[diff_mu > 0] = diff_mu[diff_mu > 0] - band[diff_mu > 0]
    if (diff_mu < 0).any():
        diff_mu[diff_mu < 0] = diff_mu[diff_mu < 0] + band[diff_mu < 0]

    diff = diff_mu.sum() / len(mu1)
    percent_diff = round((diff / avg_mu) * 100, 2)

    return percent_diff
