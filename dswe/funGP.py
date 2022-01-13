# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_features, validate_inputs
from ._GPMethods import *
from ._funGP_subroutine import *


class FunGP(object):

    def __init__(self, conf_level=0.95, limit_memory=False, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, range_seed=1):

        self.conf_level = conf_level
        self.limit_memory = limit_memory
        self.opt_method = opt_method
        self.optim_size = sample_size['optim_size']
        self.band_size = sample_size['band_size']
        self.range_seed = range_seed

    def fit(self, X, y, testX):

        self.X = X
        self.y = y
        self.testX = testX

        validate_inputs(self.X, self.y)
        validate_features(self.testX)

        optim_result = estimate_parameters(
            self.X, self.y, self.optim_size, self.range_seed, opt_method=self.opt_method, limit_memory=self.limit_memory)
        self.params = optim_result['estimated_params']
        self.diff_cov = compute_diff_conv(
            self.X, self.y, self.params, self.testX, self.band_size, self.range_seed)
        self.mu_diff = self.diff_cov['mu2'] - self.diff_cov['mu1']
        self.band = compute_conf_band(
            self.diff_cov['diff_cov_mat'], self.conf_level)

        return {'mu_diff': self.mu_diff, 'mu2': self.diff_cov['mu2'], 'mu1': self.diff_cov['mu1'],
                'band': self.band, 'conf_level': self.conf_level, 'estimated_params': self.params}
