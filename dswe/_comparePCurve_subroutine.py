# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import warnings
from sklearn.neighbors import KernelDensity


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
        return np.array([[m, n] for n in var2range for m in var1range])
#         return np.c_[var1range, var2range]
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
        diff_mu[np.abs(diff_mu) < band] = 0
    if (diff_mu > 0).any():
        diff_mu[diff_mu > 0] = diff_mu[diff_mu > 0] - band[diff_mu > 0]
    if (diff_mu < 0).any():
        diff_mu[diff_mu < 0] = diff_mu[diff_mu < 0] + band[diff_mu < 0]

    diff = diff_mu.sum() / len(mu1)
    percent_diff = round((diff / avg_mu) * 100, 2)

    return percent_diff


def compute_weighted_diff(dlist, mu1, mu2, testset, baseline):
    mixed_data = np.vstack([dlist[0], dlist[1]])

    if mixed_data.shape[1] == 1:
        var1_range = np.linspace(min(mixed_data[:, 0]), max(
            mixed_data[:, 0]), len(mixed_data)).reshape(-1, 1)
        kde1 = KernelDensity(kernel='gaussian').fit(mixed_data)
        var1_density = np.exp(kde1.score_samples(var1_range))
        var1_test = np.exp(kde1.score_samples(testset))

        prob_test = var1_test / sum(var1_test)
    else:
        var1_range = np.linspace(min(mixed_data[:, 0]), max(
            mixed_data[:, 0]), len(mixed_data)).reshape(-1, 1)
        kde1 = KernelDensity(kernel='gaussian').fit(
            mixed_data[:, 0].reshape(-1, 1))
        var1_density = np.exp(kde1.score_samples(var1_range))
        var1_test = np.exp(kde1.score_samples(testset[:, 0].reshape(-1, 1)))

        var2_range = np.linspace(min(mixed_data[:, 1]), max(
            mixed_data[:, 1]), len(mixed_data)).reshape(-1, 1)
        kde2 = KernelDensity(kernel='gaussian').fit(
            mixed_data[:, 1].reshape(-1, 1))
        var2_density = np.exp(kde2.score_samples(var2_range))
        var2_test = np.exp(kde2.score_samples(testset[:, 1].reshape(-1, 1)))

        prob_test = var1_test * var2_test / sum(var1_test * var2_test)

    diff = sum((mu2 - mu1) * (prob_test))
    if baseline == 1:
        avg_mu = sum(mu1 * prob_test)
    elif baseline == 2:
        avg_mu = sum(mu2 * prob_test)
    else:
        avg_mu = sum((mu1 + mu2) * prob_test) / 2

    # difference in result due to bandwidth value in kde for R and Python
    percent_diff = round((diff / avg_mu) * 100, 2)

    return percent_diff


def compute_weighted_stat_diff(dlist, mu1, mu2, band, testset, baseline):
    mixed_data = np.vstack([dlist[0], dlist[1]])

    if mixed_data.shape[1] == 1:
        var1_range = np.linspace(min(mixed_data[:, 0]), max(
            mixed_data[:, 0]), len(mixed_data)).reshape(-1, 1)
        kde1 = KernelDensity(kernel='gaussian').fit(mixed_data)
        var1_density = np.exp(kde1.score_samples(var1_range))
        var1_test = np.exp(kde1.score_samples(testset))

        prob_test = var1_test / sum(var1_test)
    else:
        var1_range = np.linspace(min(mixed_data[:, 0]), max(
            mixed_data[:, 0]), len(mixed_data)).reshape(-1, 1)
        kde1 = KernelDensity(kernel='gaussian').fit(
            mixed_data[:, 0].reshape(-1, 1))
        var1_density = np.exp(kde1.score_samples(var1_range))
        var1_test = np.exp(kde1.score_samples(testset[:, 0].reshape(-1, 1)))

        var2_range = np.linspace(min(mixed_data[:, 1]), max(
            mixed_data[:, 1]), len(mixed_data)).reshape(-1, 1)
        kde2 = KernelDensity(kernel='gaussian').fit(
            mixed_data[:, 1].reshape(-1, 1))
        var2_density = np.exp(kde2.score_samples(var2_range))
        var2_test = np.exp(kde2.score_samples(testset[:, 1].reshape(-1, 1)))

        prob_test = var1_test * var2_test / sum(var1_test * var2_test)

    diff_mu = mu2 - mu1
    if (np.abs(diff_mu) < band).any():
        diff_mu[np.abs(diff_mu) < band] = 0
    if (diff_mu > 0).any():
        diff_mu[diff_mu > 0] = diff_mu[diff_mu > 0] - band[diff_mu > 0]
    if (diff_mu < 0).any():
        diff_mu[diff_mu < 0] = diff_mu[diff_mu < 0] + band[diff_mu < 0]

    diff = sum(diff_mu * (prob_test))
    if baseline == 1:
        avg_mu = sum(mu1 * prob_test)
    elif baseline == 2:
        avg_mu = sum(mu2 * prob_test)
    else:
        avg_mu = sum((mu1 + mu2) * prob_test) / 2

    # difference in result due to slight difference in band values for R and Python
    percent_diff = round((diff / avg_mu) * 100, 2)

    return percent_diff


def compute_scaled_diff(ylist, mu1, mu2, nbins, baseline):
    yval = np.concatenate([ylist[0], ylist[1]])
    yval[yval < 0] = 0

    band_width = max(yval) / nbins
    start = 0
    end = max(yval) + (band_width - max(yval) % band_width)

    bins = np.arange(start, end + 1e-3, band_width)

    cum_count = [len(np.where(yval < bins[x])[0]) for x in range(len(bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = bins[dup_idx]

    cum_count = [len(np.where(mu1 < bins[x])[0])
                 for x in range(len(non_empty_bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = non_empty_bins[dup_idx]

    cum_count = [len(np.where(mu2 < bins[x])[0])
                 for x in range(len(non_empty_bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = non_empty_bins[dup_idx]

    if max(non_empty_bins) < end:
        non_empty_bins[len(non_empty_bins) - 1] = end

    total_count = len(yval)
    prob = np.array([len(np.where((yval >= non_empty_bins[x - 1]) & (yval <
                    non_empty_bins[x]))[0]) / total_count for x in range(1, len(non_empty_bins))])

    delta = mu2 - mu1

    if baseline == 1:
        mu = mu1
        delta_bin = [delta[np.where((mu1 >= non_empty_bins[x - 1]) & (
            mu1 < non_empty_bins[x]))].mean() for x in range(1, len(non_empty_bins))]
        mu_bin = [mu[np.where((mu1 >= non_empty_bins[x - 1]) & (mu1 < non_empty_bins[x]))].mean()
                  for x in range(1, len(non_empty_bins))]

    elif baseline == 2:
        mu = mu2
        delta_bin = [delta[np.where((mu2 >= non_empty_bins[x - 1]) & (
            mu2 < non_empty_bins[x]))].mean() for x in range(1, len(non_empty_bins))]
        mu_bin = [mu[np.where((mu2 >= non_empty_bins[x - 1]) & (mu2 < non_empty_bins[x]))].mean()
                  for x in range(1, len(non_empty_bins))]

    else:
        mu = 0.5 * (mu1 + mu2)
        delta_ref1 = [delta[np.where((mu1 >= non_empty_bins[x - 1]) & (
            mu1 < non_empty_bins[x]))] for x in range(1, len(non_empty_bins))]
        delta_ref2 = [delta[np.where((mu2 >= non_empty_bins[x - 1]) & (
            mu2 < non_empty_bins[x]))] for x in range(1, len(non_empty_bins))]
        delta_bin = [np.concatenate(
            [delta_ref1[x], delta_ref2[x]]).mean() for x in range(len(delta_ref1))]
        mu_ref1 = [mu[np.where((mu1 >= non_empty_bins[x - 1]) & (mu1 < non_empty_bins[x]))]
                   for x in range(1, len(non_empty_bins))]
        mu_ref2 = [mu[np.where((mu2 >= non_empty_bins[x - 1]) & (mu2 < non_empty_bins[x]))]
                   for x in range(1, len(non_empty_bins))]
        mu_bin = [np.concatenate([mu_ref1[x], mu_ref2[x]]).mean()
                  for x in range(len(mu_ref1))]

    scaled_diff = np.matmul(prob.T, delta_bin)
    percent_scaled_diff = scaled_diff * 100 / np.matmul(prob.T, mu_bin)

    return round(percent_scaled_diff, 2)


def compute_scaled_stat_diff(ylist, mu1, mu2, band, nbins, baseline):
    yval = np.concatenate([ylist[0], ylist[1]])
    yval[yval < 0] = 0

    band_width = max(yval) / nbins
    start = 0
    end = max(yval) + (band_width - max(yval) % band_width)

    bins = np.arange(start, end + 1e-3, band_width)

    cum_count = [len(np.where(yval < bins[x])[0]) for x in range(len(bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = bins[dup_idx]

    cum_count = [len(np.where(mu1 < bins[x])[0])
                 for x in range(len(non_empty_bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = non_empty_bins[dup_idx]

    cum_count = [len(np.where(mu2 < bins[x])[0])
                 for x in range(len(non_empty_bins))]
    dup_idx = [idx for idx, item in enumerate(
        cum_count) if item not in cum_count[:idx]]
    non_empty_bins = non_empty_bins[dup_idx]

    if max(non_empty_bins) < end:
        non_empty_bins[len(non_empty_bins) - 1] = end

    total_count = len(yval)
    prob = np.array([len(np.where((yval >= non_empty_bins[x - 1]) & (yval <
                    non_empty_bins[x]))[0]) / total_count for x in range(1, len(non_empty_bins))])

    delta = mu2 - mu1
    if (np.abs(delta) <= band).any():
        delta[np.abs(delta) <= band] = 0
    if (delta > 0).any():
        delta[delta > 0] = delta[delta > 0] - band[delta > 0]
    if (delta < 0).any():
        delta[delta < 0] = delta[delta < 0] + band[delta < 0]

    if baseline == 1:
        mu = mu1
        delta_bin = [delta[np.where((mu1 >= non_empty_bins[x - 1]) & (
            mu1 < non_empty_bins[x]))].mean() for x in range(1, len(non_empty_bins))]
        mu_bin = [mu[np.where((mu1 >= non_empty_bins[x - 1]) & (mu1 < non_empty_bins[x]))].mean()
                  for x in range(1, len(non_empty_bins))]

    elif baseline == 2:
        mu = mu2
        delta_bin = [delta[np.where((mu2 >= non_empty_bins[x - 1]) & (
            mu2 < non_empty_bins[x]))].mean() for x in range(1, len(non_empty_bins))]
        mu_bin = [mu[np.where((mu2 >= non_empty_bins[x - 1]) & (mu2 < non_empty_bins[x]))].mean()
                  for x in range(1, len(non_empty_bins))]

    else:
        mu = 0.5 * (mu1 + mu2)
        delta_ref1 = [delta[np.where((mu1 >= non_empty_bins[x - 1]) & (
            mu1 < non_empty_bins[x]))] for x in range(1, len(non_empty_bins))]
        delta_ref2 = [delta[np.where((mu2 >= non_empty_bins[x - 1]) & (
            mu2 < non_empty_bins[x]))] for x in range(1, len(non_empty_bins))]
        delta_bin = [np.concatenate(
            [delta_ref1[x], delta_ref2[x]]).mean() for x in range(len(delta_ref1))]
        mu_ref1 = [mu[np.where((mu1 >= non_empty_bins[x - 1]) & (mu1 < non_empty_bins[x]))]
                   for x in range(1, len(non_empty_bins))]
        mu_ref2 = [mu[np.where((mu2 >= non_empty_bins[x - 1]) & (mu2 < non_empty_bins[x]))]
                   for x in range(1, len(non_empty_bins))]
        mu_bin = [np.concatenate([mu_ref1[x], mu_ref2[x]]).mean()
                  for x in range(len(mu_ref1))]

    scaled_diff = np.matmul(prob.T, delta_bin)
    percent_scaled_diff = scaled_diff * 100 / np.matmul(prob.T, mu_bin)

    return round(percent_scaled_diff, 2)


def compute_ratio(dlist1, dlist2):
    comb_list1 = np.vstack([dlist1[0], dlist1[1]])
    comb_list2 = np.vstack([dlist2[0], dlist2[1]])
    ratio_col = (comb_list2.max(axis=0) - comb_list2.min(axis=0)) / \
        (comb_list1.max(axis=0) - comb_list2.min(axis=0))

    if dlist1[0].shape[1] == 1:
        return {'ratio_col1': ratio_col}
    else:
        return {'ratio_col1': ratio_col[0], 'ratio_col2': ratio_col[1]}
