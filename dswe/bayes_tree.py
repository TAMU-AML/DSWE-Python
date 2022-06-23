# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bartpy.sklearnmodel import SklearnModel


class BayesTreePowerCurve(object):

    """
    Parameters
    ----------
    X_train: np.ndarray or pd.DataFrame
        A matrix or dataframe of input variable values in the training dataset.

    y_train: np.array
        A numeric array for response values in the training dataset.

    n_trees: int
        Number of trees to use. An integer greater than 0.

    """

    def __init__(self, X_train, y_train, n_trees=200):

        if not (isinstance(X_train, list) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series) or isinstance(X_train, np.ndarray)):
            raise ValueError(
                "The X_train should be either a list or numpy array or dataframe.")

        if not (isinstance(y_train, list) or isinstance(y_train, np.ndarray)) or isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            raise ValueError(
                "The target data should be either a list or numpy array or dataframe.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "The X_train and y_train should have same number of data points.")

        if not isinstance(n_trees, int):
            raise ValueError("The number of trees must be an integer value.")

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.n_trees = n_trees

        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        # scale the features
        self.scale_features = StandardScaler()
        self.scale_features.fit(self.X_train)
        self.X_train = self.scale_features.transform(self.X_train)

        self.is_discrete = False

        if (self.y_train == self.y_train.astype(int)).all():
            # target values are discrete
            self.y_train = self.y_train.astype(int)
            self.is_discrete = True
        else:
            # target values are continuous
            self.scale_target = StandardScaler()    # scale the target
            self.scale_target.fit(self.y_train.reshape(-1, 1))
            self.y_train = self.scale_target.transform(
                self.y_train.reshape(-1, 1)).squeeze()

        self.model = SklearnModel(self.n_trees)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        """
        Parameters
        ----------
        X_test: np.ndarray or pd.DataFrame
            A matrix or dataframe of test input variable values to compute predictions.

        Returns
        -------
        np.array
            A numeric array for predictions at the data points in X_test.

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

        X_test = self.scale_features.transform(X_test)

        predictions = self.model.predict(X_test)
        if not self.is_discrete:
            predictions = self.scale_target.inverse_transform(
                predictions.reshape(-1, 1)).squeeze()

        return predictions
