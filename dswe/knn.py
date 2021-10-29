# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from ._knn_subroutine import compute_best_k


class KNNPowerCurve(object):

    """
    k-nearest neighbors regression model.


    All the parameters are fine-tuned automatically using GridSearch.
    No need to explicitly pass anything.

    Parameters
    ----------

    weights : {'auto', 'distance'}
            - 'uniform' : uniform weights.  All points in each neighborhood
            are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
            in this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.

    algorithms : Algorithm used to compute the nearest neighbors.
                - 'auto' will attempt to decide the most appropriate algorithm
                based on the values passed to 'fit' method.

    """

    def __init__(
            self,
            algorithms=['auto'],
            weights=['uniform', 'distance']):

        self.algorithms = algorithms
        self.weights = weights

    def validate_inputs(self, X, y):
        """Validates the inputs to KNNPowerCurve."""
        if not (isinstance(X, list) or isinstance(pd.DataFrame(X), pd.DataFrame) or isinstance(np.array(X), np.ndarray)):
            raise ValueError(
                "The features data should be either of list or numpy array or dataframe.")

        if not (isinstance(y, list) or isinstance(pd.DataFrame(y), pd.DataFrame) or isinstance(np.array(y), np.ndarray)):
            raise ValueError(
                "The target data should be either of list or numpy array or dataframe.")

        if np.isnan(np.array(X)).any() or np.isnan(np.array(y)).any():
            raise ValueError("The data should not contains any null value.")

        if np.isinf(np.array(X)).any() or np.isinf(np.array(y)).any():
            raise ValueError("The data must not have any infinite value.")

        if not np.isfinite(np.array(X)).any() or not np.isfinite(np.array(y)).any():
            raise ValueError(
                "The data should have only numeric values.")

    def fit(self, X, y, subset_selection=False):
        """Fit the KNNPowerCurve from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Return
        ------
        self : The fitted KNNPowerCurve.
        """

        self.validate_inputs(X, y)

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
            parameters = {'algorithm': self.algorithms,
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
        """Predict the target for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
           Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
           Predicted target values.

        """

        normlized_X = (X - self.scaler_min) / \
            (self.scaler_max - self.scaler_min)

        y_pred = KNeighborsRegressor(n_neighbors=self.best_k).fit(
            self.normalized_X, self.y).predict(normlized_X)

        return y_pred

    def update(self, X, y):
        """Update the model when new training dataset will arrive.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            New target values.

        Return
        ------
        self : The fitted KNNPowerCurve.
        """

        self.validate_inputs(X, y)

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
