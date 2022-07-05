# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from ._AMK_subroutine import *


class AMK(object):

    """
    References
    ----------
    Lee, Ding, Genton, and Xie, 2015, “Power curve estimation with multivariate environmental factors
    for inland and offshore wind farms,” Journal of the American Statistical Association, Vol. 110, pp.
    56-67, DOI:10.1080/01621459.2014.977385.

    Parameters
    ----------
    X_train: np.ndarray or pd.DataFrame
        A matrix or dataframe of input variable values in the training dataset.

    y_train: np.array
        A numeric array for response values in the training dataset.

    X_test: np.ndarray or pd.DataFrame
        A matrix or dataframe of test input variable values to compute predictions.

    bw: string or int
        A numeric array or a character input for bandwidth. If character, bandwidth
        computed internally; the input should be either 'dpi' or 'dpi_gap'. 
        Default value is 'dpi_gap'.

    n_multi_cov: int
        An integer or a character input specifying the number of multiplicative covariates
        in each additive term. Default is 3 (same as Lee et al., 2015). The character
        inputs can be: 'all' for a completely multiplicative model, or 'none' for a
        completely additive model. Ignored if the number of covariates is 1.

    fixed_cov: list
        An integer list specifying the fixed covariates column number(s).
        Ignored if n_multi_cov is set to 'all' or 'none' or if the number of covariates is less than 3.
        Default value is [0,1].

    cir_cov: list or int
        A list specifying the circular covariates column number(s) in X_train,
        An integer when only one circular covariates present. 
        Default value is None.

    Returns
    -------
    AMK
        self with trained parameter values. \n
        - predictions: stored numeric array of model output at the data points in X_test.
    """

    def __init__(self, X_train, y_train, X_test, bw='dpi', n_multi_cov=3, fixed_cov=[0, 1], cir_cov=None):

        if not (isinstance(X_train, list) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series) or isinstance(X_train, np.ndarray)):
            raise ValueError(
                "The X_train should be either a list or numpy array or dataframe.")
        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        if len(X_train.shape) > 1:
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    "The number of features in train and test set must be same.")

        if not (isinstance(y_train, list) or isinstance(y_train, np.ndarray)) or isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            raise ValueError(
                "The target data should be either a list or numpy array or dataframe.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "The X_train and y_train should have same number of data points.")

        if len(X_train.shape) == 2:
            ncov = X_train.shape[1]
        else:
            ncov = 1

        if not (isinstance(bw, list) or isinstance(bw, np.ndarray)):
            if bw not in ['dpi', 'dpi_gap']:
                raise ValueError(
                    "The bw must a list or an array or set to 'dpi' or 'dpi_gap'.")
        elif len(bw) != ncov:
            raise ValueError(
                "The length of bw must be same as the number of covariates.")

        if type(n_multi_cov) != int:
            if n_multi_cov not in ['all', 'none']:
                raise ValueError(
                    "The n_multi_cov must be set to 'all' or 'none' or an integer.")

        if ncov == 1:
            n_multi_cov = 'all'
        elif ncov == 2:
            if n_multi_cov != 'none':
                n_multi_cov = 'all'

        if n_multi_cov not in ['all', 'none']:
            if n_multi_cov < 1 or n_multi_cov > ncov:
                raise ValueError(
                    "if n_multi_cov is not set to 'all' or 'none', then it must be set to an integer greater than 1, and less than or equal to the number of covariates.")
            elif n_multi_cov == ncov:
                n_multi_cov = 'all'
                fixed_cov = None
            elif n_multi_cov < ncov:
                if fixed_cov is not None:
                    if not (isinstance(fixed_cov, list) or isinstance(fixed_cov, np.ndarray)):
                        raise ValueError(
                            "The fixed_cov should either be a list or an array or set to None.")
                    elif len(list(set(fixed_cov).intersection(list(range(ncov))))) != len(fixed_cov):
                        raise ValueError(
                            "Any or all the values in fixed_cov exceeds the number of columns in X_train.")
                    elif len(fixed_cov) >= n_multi_cov:
                        raise ValueError(
                            "The fixed_cov should be less than n_multi_cov.")
        elif n_multi_cov in ['all', 'none']:
            fixed_cov = None

        if cir_cov is not None:
            if not (isinstance(cir_cov, list) or isinstance(cir_cov, np.ndarray) or type(cir_cov) == int):
                raise ValueError(
                    "The circ_cov should be a list or 1d-array or single integer value or set to None.")
            if type(cir_cov) == int:
                cir_cov = [cir_cov]
            elif len(list(set(cir_cov).intersection(list(range(ncov))))) != len(cir_cov):
                raise ValueError(
                    "Any or all the values in cir_cov exceeds the number of columns in X_train.")

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.bw = bw
        self.n_multi_cov = n_multi_cov
        self.fixed_cov = fixed_cov
        self.cir_cov = cir_cov

        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)
            self.X_test = self.X_test.reshape(-1, 1)

        self.predictions = kern_pred(X_train, y_train, X_test, bw,
                                     n_multi_cov, fixed_cov, cir_cov)
