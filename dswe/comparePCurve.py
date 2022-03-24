# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from ._comparePCurve_subroutine import *
from .funGP import *
from .covmatch import *


class ComparePCurve(object):

    def __init__(self, Xlist, ylist, testcol, testX=None, circ_pos=None, thresh=0.2, conf_level=0.95, grid_size=[50, 50],
                 power_bins=15, baseline=1, limit_memory=True, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, rng_seed=1):

        validate_matching(Xlist, ylist)

        if not (isinstance(testcol, list) or isinstance(testcol, np.ndarray)):
            raise ValueError(
                "The testcol must be provided in a list or 1d-array.")
        if len(testcol) > 2:
            raise ValueError("Maximum two columns to be used.")

        if circ_pos:
            if not (isinstance(circ_pos, list) or isinstance(circ_pos, np.ndarray)):
                raise ValueError(
                    "The circ_pos must be provided in a list or 1d-array.")

        if isinstance(thresh, list) or isinstance(thresh, np.ndarray):
            if len(thresh) > 0:
                if len(thresh) != Xlist[0].shape[1]:
                    raise ValueError(
                        "The thresh must be a single value, or list or 1d array with weight for each covariate.")

        if type(conf_level) != int and type(conf_level) != float or conf_level < 0 or conf_level > 1:
            raise ValueError(
                "The conf_level be a numeric value between 0 and 1")

        if not (isinstance(grid_size, list) or isinstance(grid_size, np.ndarray)):
            raise ValueError(
                "The grid_size must be provided in a list or 1d-array.")
        elif len(grid_size) != len(testcol):
            raise ValueError(
                "The length of grid_size should be equal to two when length of testCol is equal to two.")

        if baseline not in [0, 1, 2]:
            raise ValueError("The basline must be an integer between 0 and 2.")

        if type(limit_memory) != type(True):
            raise ValueError("The limit_memory must be either True or False.")

        if limit_memory:
            if not isinstance(sample_size, dict):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")
            if not set(['optim_size', 'band_size']) == set(list(sample_size.keys())):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")

        if type(rng_seed) != int:
            raise ValueError("The range seed must be a single integer value.")

        if opt_method not in ['L-BFGS-B', 'BFGS']:
            raise ValueError("The opt_method must be 'L-BFGS-B' or 'BFGS'.")

        self.conf_level = conf_level
        self.opt_method = opt_method
        self.rng_seed = rng_seed

        self.Xlist = Xlist
        self.ylist = ylist
        self.Xlist[0] = np.array(self.Xlist[0])
        self.Xlist[1] = np.array(self.Xlist[1])
        self.ylist[0] = np.array(self.ylist[0]).reshape(-1, 1)
        self.ylist[1] = np.array(self.ylist[1]).reshape(-1, 1)

        self.testX = testX

        result_matching = CovMatch(self.Xlist, self.ylist, circ_pos, thresh)
        self.matched_data_X = result_matching.matched_data_X
        self.matched_data_y = result_matching.matched_data_y

        if self.testX is None:
            self.testX = generate_test_set(
                self.matched_data_X, testcol, grid_size)

        _mdata_X = [self.matched_data_X[0][:, testcol],
                    self.matched_data_X[1][:, testcol]]
        _mdata_y = [self.matched_data_y[0], self.matched_data_y[1]]
        result_GP = FunGP(_mdata_X, _mdata_y, self.testX, self.conf_level,
                          limit_memory, self.opt_method, sample_size=sample_size, rng_seed=self.rng_seed)

        self.mu1 = result_GP.mu1
        self.mu2 = result_GP.mu2
        self.mu_diff = result_GP.mu_diff
        self.band = result_GP.band
        self.estimated_params = result_GP.params

        self.weighted_diff = compute_weighted_diff(
            self.Xlist, self.mu1, self.mu2, self.testX, testcol, baseline)

        self.weighted_stat_diff = compute_weighted_stat_diff(
            self.Xlist, self.mu1, self.mu2, self.band, self.testX, testcol, baseline)

        self.scaled_diff = compute_scaled_diff(
            self.ylist, self.mu1, self.mu2, power_bins, baseline)

        self.scaled_stat_diff = compute_scaled_stat_diff(
            self.ylist, self.mu1, self.mu2, self.band, power_bins, baseline)

        self.unweighted_diff = compute_diff(self.mu1, self.mu2, baseline)

        self.unweighted_stat_diff = compute_stat_diff(
            self.mu1, self.mu2, self.band, baseline)

        self.reduction_ratio = compute_ratio(
            self.Xlist, self.matched_data_X, testcol)

    def compute_weighted_difference(self, weights, baseline=1, stat_diff=False):

        weights = np.array(weights)

        if not (isinstance(weights, list) or isinstance(weights, np.ndarray)):
            raise ValueError("The weights must be a numeric vector.")

        if len(weights) != len(self.mu_diff):
            raise ValueError(
                "The length of weights vector must be equal to the length of mu_diff which has calculated in creating this class instance")
        if abs(weights.sum() - 1) >= 1e-3:
            raise ValueError(
                "The weights must sume to 1 for being a valid weights.")

        if type(stat_diff) != type(True):
            raise ValueError("The stat_diff must be either True or False.")

        if baseline == 1:
            return compute_weighted_diff_extern(self.mu_diff, weights, self.mu1, stat_diff, self.band)
        else:
            return compute_weighted_diff_extern(self.mu_diff, weights, self.mu2, stat_diff, self.band)
