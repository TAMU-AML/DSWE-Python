# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import math
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from ._knn_subroutine import compute_best_k


class KNNPowerCurve(object):

    def __init__(
            self,
            algorithms=['auto', 'kd_tree', 'ball_tree'],
            weights=['uniform']):

        self.algorithms = algorithms
        self.weights = weights

    def check_data_format(self, X, y):
        try:
            not isinstance(X, list) or not isinstance(pd.DataFrame(
                X), pd.DataFrame) or isinstance(np.array(X), np.ndarray)
        except:
            print(
                "The train features data should be either of list or numpy array or dataframe.")

        try:
            not isinstance(y, list) or not isinstance(pd.DataFrame(
                y), pd.DataFrame) or isinstance(np.array(y), np.ndarray)
        except:
            print(
                "The train target data should be either of list or numpy array or dataframe.")

    def fit(self, X, y, subset_selection=False):
        self.check_data_format(X, y)
        X = np.array(X)
        y = np.array(y)

        normalized_X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        range_k = np.linspace(5, 50, 10, dtype=int)

        if not subset_selection:
            result = compute_best_k(X, y, range_k)

            knn = KNeighborsRegressor(n_neighbors=result['best_k'])
            parameters = {'algorithm': self.algorithms,
                          'weights': self.weights}
            regressor = GridSearchCV(knn, parameters)
            regressor.fit(normalized_X, y)
            mae = np.mean(abs(regressor.predict(normalized_X) - y))

            return {'best_k': result['best_k'], 'RMSE': result['best_rmse'],
                    'MAE': mae, 'X': X, 'y': y}

        else:
            print("Subset selection choice is not available yet.")

    def predict(self, knn_model, X):
        train_X = knn_model['X']
        train_y = knn_model['y']
        best_k = knn_model['best_k']

        normlized_train_X = (train_X - train_X.min(axis=0)) / \
            (train_X.max(axis=0) - train_X.min(axis=0))
        normlized_test_X = (X - X.min(axis=0)) / \
            (X.max(axis=0) - X.min(axis=0))

        prediction = KNeighborsRegressor(n_neighbors=best_k).fit(
            normlized_train_X, train_y).predict(normlized_test_X)

        return prediction

    def update(self, knn_model, X, y):
        old_X = knn_model['X']
        old_y = knn_model['y']
        old_best_k = knn_model['best_k']

        new_X = np.concatenate([old_X, X])
        new_y = np.concatenate([old_y, y])

        normalized_X = (new_X - new_X.min(axis=0)) / \
            (new_X.max(axis=0) - new_X.min(axis=0))

        ubk = 1.2
        lbk = 0.8
        interval_k = 5
        max_k = math.ceil(ubk * old_best_k)
        max_k = max_k + (interval_k - (max_k // interval_k))
        min_k = math.floor(lbk * old_best_k)
        min_k = min_k - (max_k // interval_k)
        range_k = np.linspace(min_k, max_k, interval_k, dtype=int)

        result = compute_best_k(normalized_X, new_y, range_k)

        result = {'best_K': result['best_k'], 'X': normalized_X, 'y': new_y}

        return result
