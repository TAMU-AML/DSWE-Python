# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from _comparePCurve_subroutine import *
from funGP import *
from covmatch import *


class ComparePCurve(object):

    def __init__(self, Xlist, ylist, testcol, testX=None, circ_pos=None, thresh=0.1, conf_level=0.95, grid_size=[20, 20],
                 power_bins=15, baseline=1, limit_memory=True, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, range_seed=1):

        self.conf_level = conf_level
        self.opt_method = opt_method
        self.range_seed = range_seed

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

        if not self.testX:
            self.testX = generate_test_set(
                self.matched_data_X, testcol, grid_size)

        _mdata_X = [self.matched_data_X[0][:, testcol],
                    self.matched_data_X[1][:, testcol]]
        _mdata_y = [self.matched_data_y[0], self.matched_data_y[1]]
        result_GP = FunGP(_mdata_X, _mdata_y, self.testX, self.conf_level,
                          limit_memory, self.opt_method, sample_size=sample_size, range_seed=self.range_seed)

        self.mu1 = result_GP.mu1
        self.mu2 = result_GP.mu2
        self.mu_diff = result_GP.mu_diff
        self.band = result_GP.band
        self.estimated_params = result_GP.params

        self.weight_diff = compute_weighted_diff(
            self.Xlist, self.mu1, self.mu2, self.testX, testcol, baseline)

        self.weight_stat_diff = compute_weighted_stat_diff(
            self.Xlist, self.mu1, self.mu2, self.band, self.testX, testcol, baseline)

        self.scale_diff = compute_scaled_diff(
            self.ylist, self.mu1, self.mu2, power_bins, baseline)

        self.scale_stat_diff = compute_scaled_stat_diff(
            self.ylist, self.mu1, self.mu2, self.band, power_bins, baseline)

        self.unweight_diff = compute_diff(self.mu1, self.mu2, baseline)

        self.unweight_stat_diff = compute_stat_diff(
            self.mu1, self.mu2, self.band, baseline)

        self.reduction_ratio = compute_ratio(
            self.Xlist, self.matched_data_X, testcol)
