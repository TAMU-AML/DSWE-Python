# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from .utils import validate_inputs
from ._temp_GP_cpp import *
from ._tempGP_subroutine import *


class TempGP(object):

    def __init__(self, opt_method):
        self.opt_method = opt_method

    def fit(self, X, y, T=None):
        validate_inputs(X, y)
        if T:
            validate_inputs(X, T)

        self.X = np.array(X)
        self.y = np.array(y)
        if T:
            self.T = np.array(T)

        self.thinning_number = compute_thinning_number(self.X, self.y)
        self.databins = create_thinned_bins(
            self.X, self.y, self.thinning_number)
        self.optim_result = estimate_binned_params(
            self.databins, self.opt_method)
        self.weighted_y = compute_weighted_y(
            self.X, self.y, self.optim_result['estimated_params'])
        self.model_F = {'train_X': self.X, 'train_y': self.y,
                        'weighted_y': self.weighted_y}

        self.train_residual = self.y - predict_GP(self.X, self.weighted_y, self.X,
                                                  self.optim_result['estimated_params'])

        self.model_G = {'residual': self.train_residual, 'time_index': self.T}

        return self

    def predict(self, X, test_T=None, train_T=None):

        X = np.array(X)

        if test_T:
            test_T = np.array(test_T)
        if train_T:
            train_T = np.array(train_T)
            self.model_G['time_index'] = train_T

        pred_F = predict_GP(self.X, self.weighted_y, X,
                            self.optim_result['estimated_params'])

        if not test_T:
            return pred_F
        else:
            return pred_F  # ++ compute local function

    def update(self, X, y, T=None, replace=True, update_model_F=False):
