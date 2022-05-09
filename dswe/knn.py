# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from .utils import validate_inputs
from ._knn_subroutine import compute_best_k


class KNNPowerCurve(object):

    """
    All the parameters are fine-tuned automatically using GridSearch.
    No need to explicitly pass anything.

    Parameters
    ----------
    algorithm: 
        Algorithm used to compute the nearest neighbors.
        'auto' attempt to decide the most appropriate algorithm based on the values passed to 'fit' method.

    weights: 
        Weight function used in prediction. Can take either 'uniform' or 'distance'.
        'uniform' means uniform weights i.e., all points in each neighborhood are weighted equally.
        'distance' means weight points by the inverse of their distance.
    """

    def __init__(self, algorithm=['auto'], weights=['uniform', 'distance']):

        self.algorithm = algorithm
        self.weights = weights

    def fit(self, X_train, y_train, subset_selection=False):
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

        if not subset_selection:
            result = compute_best_k(
                self.normalized_X_train, self.y_train, range_k)

            knn = KNeighborsRegressor(n_neighbors=result['best_k'])
            parameters = {'algorithm': self.algorithm,
                          'weights': self.weights}
            regressor = GridSearchCV(knn, parameters)
            regressor.fit(self.normalized_X_train, self.y_train)
            mae = np.mean(abs(regressor.predict(
                self.normalized_X_train) - self.y_train))

            self.best_k = result['best_k']
            self.best_rmse = result['best_rmse']
            self.mae = mae

            self.model = KNeighborsRegressor(n_neighbors=self.best_k).fit(
                self.normalized_X_train, self.y_train)

            return self

        else:
            print("Subset selection choice is not available yet.")

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

        X_test = np.array(X_test)

        normlized_X_test = (X_test - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

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
