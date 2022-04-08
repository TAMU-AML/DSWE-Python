# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd


class AMK(object):

    def __init__(self, X_train, y_train, X_test, bw='dpi_gap', n_multi_cov=3, fixed_cov=[1, 2], cir_cov=None):

        if not (isinstance(X_train, list) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray)):
            raise ValueError(
                "The X_train should be either a list or numpy array or dataframe.")
        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                "The number of features in train and test set must be same.")

        if not (isinstance(y_train, list) or isinstance(y_train, np.ndarray)):
            raise ValueError(
                "The target data should be either a list or numpy array.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "The X_train and y_train should have same number of data points.")

        n_cov = X_train.shape[1]

        if not (isinstance(bw, list) or isinstance(bw, np.ndarray)):
            if bw not in ['dpi', 'dpi_gap']:
                raise ValueError(
                    "The bw must a list or an array or set to 'dpi' or 'dpi_gap'.")
        elif len(bw) != n_cov:
            raise ValueError(
                "The length of bw must be same as the number of covariates.")

        if type(n_multi_cov) != int:
            if n_multi_cov not in ['all', 'none']:
                raise ValueError(
                    "The n_multi_cov must be set to 'all' or 'none' or an integer.")

        if n_cov == 1:
            n_multi_cov = 'all'
        elif n_cov == 2:
            if n_multi_cov != 'none':
                n_multi_cov = 'all'

        if n_multi_cov not in ['all', 'none']:
            if n_multi_cov < 1 and n_multi_cov > n_cov:
                raise ValueError(
                    "if n_multi_cov is not set to 'all' or 'none', then it must be set to an integer greater than 1, and less than or equal to the number of covariates.")
            elif n_multi_cov == n_cov:
                n_multi_cov = 'all'
                fixed_cov = None
            elif n_multi_cov < n_cov:
                if fixed_cov is not None:
                    if not (isinstance(fixed_cov, list) or isinstance(fixed_cov, np.ndarray)):
                        raise ValueError(
                            "The fixed_cov should either be a list or an array or set to None.")
                    elif len(list(set(fixed_cov).intersection(list(range(n_cov))))) != len(fixed_cov):
                        raise ValueError(
                            "Any or all the values in fixed_cov exceeds the number of columns in X_train.")
                    elif len(fixed_cov) >= n_multi_cov:
                        raise ValueError(
                            "The fixed_cov should be less than n_multi_cov.")
        elif n_multi_cov in ['all', 'none']:
            fixed_cov = None

        if cir_cov is not None:
            if not (isinstance(cir_cov, list) or isinstance(cir_cov, np.ndarray)):
                raise ValueError(
                    "The cir_cov should be a list or an array or set to None.")
            elif len(list(set(cir_cov).intersection(list(range(n_cov))))) != len(cir_cov):
                raise ValueError(
                    "Any or all the values in cir_cov exceeds the number of columns in X_train.")

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.bw = bw
        self.n_multi_cov = n_multi_cov
        self.fixed_cov = fixed_cov
        self.cir_cov = cir_cov

        return self
