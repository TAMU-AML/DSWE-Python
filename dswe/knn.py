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

    def fit(self, X, y, subset_selection=False):
        """
        Parameters
        ----------
        X: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the training dataset.

        y: np.array
            A numeric array for response values in the training dataset.

        Returns
        -------
        KNNPowerCurve
            self with trained parameter values.
        """

        validate_inputs(X, y)

        self.X = np.array(X)
        self.y = np.array(y)

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        self.scaler_min = self.X.min(axis=0)
        self.scaler_max = self.X.max(axis=0)

        self.normalized_X = (self.X - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)
        range_k = np.linspace(5, 50, 10, dtype=int)

        if not subset_selection:
            result = compute_best_k(self.normalized_X, self.y, range_k)

            knn = KNeighborsRegressor(n_neighbors=result['best_k'])
            parameters = {'algorithm': self.algorithm,
                          'weights': self.weights}
            regressor = GridSearchCV(knn, parameters)
            regressor.fit(self.normalized_X, self.y)
            mae = np.mean(abs(regressor.predict(self.normalized_X) - self.y))

            self.best_k = result['best_k']
            self.best_rmse = result['best_rmse']
            self.mae = mae

            return self

        else:
            print("Subset selection choice is not available yet.")

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the test dataset.

        Returns
        -------
        np.array
            A numeric array for predictions at the data points in the test dataset.
        """

        normlized_X = (X - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

        y_pred = KNeighborsRegressor(n_neighbors=self.best_k).fit(
            self.normalized_X, self.y).predict(normlized_X)

        return y_pred

    def update(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the new added dataset.

        y: np.array
            A numeric array for response values in the new added dataset.

        Returns
        -------
        KNNPowerCurve
            self with updated trained parameter values.
        """

        validate_inputs(X, y)

        if X.shape[0] <= self.X.shape[0]:
            self.X = np.concatenate([self.X[X.shape[0]:], X])
            self.y = np.concatenate([self.y[y.shape[0]:], y])
        else:
            print("Please run fit function again.")

        self.scaler_min = self.X.min(axis=0)
        self.scaler_max = self.X.max(axis=0)

        self.normalized_X = (self.X - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

        ubk = 1.2
        lbk = 0.8
        interval_k = 5
        max_k = math.ceil(ubk * self.best_k)
        max_k = max_k + (interval_k - (max_k // interval_k))
        min_k = math.floor(lbk * self.best_k)
        min_k = min_k - (max_k // interval_k)
        range_k = np.linspace(min_k, max_k, interval_k, dtype=int)

        result = compute_best_k(self.normalized_X, self.y, range_k)
        self.best_k = result['best_k']

        return self
