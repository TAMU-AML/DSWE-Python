# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsRegressor
from .utils import validate_inputs
from ._knn_subroutine import compute_best_k, compute_best_subset


class KNNPowerCurve(object):

    """
    Parameters
    ----------
    algorithm: list
        Algorithm used to compute the nearest neighbors.
        'auto' attempt to decide the most appropriate algorithm based on the values passed to 'fit' method.
        Default is 'auto'.

    weights: list
        Weight function used in prediction. Can take either 'uniform' or 'distance'.
        'uniform' means uniform weights i.e., all points in each neighborhood are weighted equally.
        'distance' means weight points by the inverse of their distance.
        Default is 'uniform'.

    subset_selection: bool
        A boolean (True/False) to select the best feature columns.
        Default is set to False.

    """

    def __init__(self, algorithm='auto', weights='uniform', subset_selection=False):

        self.algorithm = algorithm
        self.weights = weights
        self.subset_selection = subset_selection

    def fit(self, X_train, y_train):
        """
        Parameters
        ----------
        X_train: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the training dataset.

        y_train: np.array
            A numeric array for response values in the training dataset.

        Returns
        -------
        KNNPowerCurve
            self with trained parameter values.
        """

        validate_inputs(X_train, y_train)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        self.scaler_min = self.X_train.min(axis=0)
        self.scaler_max = self.X_train.max(axis=0)

        self.normalized_X_train = (self.X_train - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)
        range_k = np.linspace(5, 50, 10, dtype=int)

        if not self.subset_selection:
            result = compute_best_k(
                self.normalized_X_train, self.y_train, range_k)
            self.best_k = result['best_k']

            regressor = KNeighborsRegressor(
                n_neighbors=self.best_k, algorithm=self.algorithm, weights=self.weights)
            regressor.fit(self.normalized_X_train, self.y_train)
            mae = np.mean(abs(regressor.predict(
                self.normalized_X_train) - self.y_train))

            self.best_rmse = result['best_rmse']
            self.mae = mae

            self.model = KNeighborsRegressor(n_neighbors=self.best_k).fit(
                self.normalized_X_train, self.y_train)

        else:
            result = compute_best_subset(
                self.normalized_X_train, self.y_train, range_k)
            self.best_k = result['best_k']
            self.best_subset = result['best_subset']

            regressor = KNeighborsRegressor(
                n_neighbors=self.best_k, algorithm=self.algorithm, weights=self.weights)
            regressor.fit(
                self.normalized_X_train[:, self.best_subset], self.y_train)
            mae = np.mean(abs(regressor.predict(
                self.normalized_X_train[:, self.best_subset]) - self.y_train))

            self.best_rmse = result['best_rmse']
            self.mae = mae

            self.model = KNeighborsRegressor(n_neighbors=self.best_k).fit(
                self.normalized_X_train, self.y_train)

        return self

    def predict(self, X_test):
        """
        Parameters
        ----------
        X_test: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the test dataset.

        Returns
        -------
        np.array
            A numeric array for predictions at the data points in the test dataset.
        """
        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        if len(self.X_train.shape) > 1:
            if X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    "The number of features in train and test set must be same.")

        normlized_X_test = (X_test - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

        if self.subset_selection:
            normlized_X_test = normlized_X_test[:, self.best_subset]

        y_pred = self.model.predict(normlized_X_test)

        return y_pred

    def update(self, X_update, y_update):
        """
        Parameters
        ----------
        X_update: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the new added dataset.

        y_update: np.array
            A numeric array for response values in the new added dataset.

        Returns
        -------
        KNNPowerCurve
            self with updated trained parameter values.
        """

        validate_inputs(X_update, y_update)

        X_update = np.array(X_update)
        y_update = np.array(y_update)

        if len(X_update.shape) == 1:
            X_update = X_update.reshape(-1, 1)

        if len(self.X_train.shape) > 1:
            if X_update.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    "The number of features in train and update set must be same.")

        if X_update.shape[0] <= self.X_train.shape[0]:
            self.X_train = np.concatenate(
                [self.X_train[X_update.shape[0]:], X_update])
            self.y_train = np.concatenate(
                [self.y_train[y_update.shape[0]:], y_update])
        else:
            print("Please run fit function again.")

        self.scaler_min = self.X_train.min(axis=0)
        self.scaler_max = self.X_train.max(axis=0)

        self.normalized_X_train = (self.X_train - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

        if self.subset_selection:
            self.normalized_X_train = self.normalized_X_train[:,
                                                              self.best_subset]

        ubk = 1.2
        lbk = 0.8
        interval_k = 5
        max_k = math.ceil(ubk * self.best_k)
        max_k = max_k + (interval_k - (max_k // interval_k))
        min_k = math.floor(lbk * self.best_k)
        min_k = min_k - (max_k // interval_k)
        range_k = np.linspace(min_k, max_k, interval_k, dtype=int)

        result = compute_best_k(self.normalized_X_train, self.y_train, range_k)
        self.best_k = result['best_k']

        self.model = KNeighborsRegressor(n_neighbors=self.best_k).fit(
            self.normalized_X_train, self.y_train)

        return self
