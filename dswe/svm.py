# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler


class SVMPowerCurve(object):

    """
    Support Vector Machine (SVM) based power curve modelling.

    Parameters
    ----------
    X_train : A matrix or dataframe of input variable values in the training dataset.

    y_train : A numeric array for response values in the training dataset.

    kernel : Kernel type to be used in the algorithm. Default is 'rbf' else can be 'linear', 'poly', 'sigmoid'. 
             'poly' mean polynomial and 'rbf' means radial basis function.

    degree : Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

    gamma : Kernel coefficient for 'poly', 'radial' and 'sigmoid'. Can take 'scale' or 'auto' or float value.
            If 'scale' (default), the gamma value is 1/(number_of_features*variance_of_X_train).
            If 'auto', the gamma value is 1/number_of_features.

    C : Regularization parameter. The strength of the regularization is inversely proportional to C. 
        Must be strictly positive.

    """

    def __init__(self, X_train, y_train, kernel='rbf', degree=3, gamma='scale', C=1.0):

        if not (isinstance(X_train, list) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray)):
            raise ValueError(
                "The X_train should be either a list or numpy array or dataframe.")

        if not (isinstance(y_train, list) or isinstance(y_train, np.ndarray)) or isinstance(y_train, pd.DataFrame):
            raise ValueError(
                "The target data should be either a list or numpy array or dataframe.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "The X_train and y_train should have same number of data points.")

        if isinstance(kernel, str):
            if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
                raise ValueError(
                    "The kernel can only take followings as input: linear, radial, polynomial and sigmoid.")
        else:
            raise ValueError("The kernel can only take string input.")

        if not isinstance(degree, int):
            raise ValueError("The degree must be an integer value.")

        if not (isinstance(gamma, int) or isinstance(gamma, float)):
            if gamma not in ['scale', 'auto']:
                raise ValueError(
                    "The gamma must be set to 'scale' or 'auto' or a numeric value.")

        if not (isinstance(C, int) or isinstance(C, float)) and C > 0:
            raise ValueError("The C must be a numeric value greater than 0.")

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.C = C

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

            if self.kernel == 'linear':
                self.model = SVC(kernel=self.kernel, C=self.C)
            elif self.kernel == 'poly':
                self.model = SVC(kernel=self.kernel,
                                 degree=self.degree, gamma=self.gamma, C=self.C)
            else:
                self.model = SVC(kernel=self.kernel,
                                 gamma=self.gamma, C=self.C)
            self.model.fit(self.X_train, self.y_train)
        else:
            # target values are continuous
            self.scale_target = StandardScaler()    # scale the target
            self.scale_target.fit(self.y_train.reshape(-1, 1))
            self.y_train = self.scale_target.transform(
                self.y_train.reshape(-1, 1)).squeeze()

            if self.kernel == 'linear':
                self.model = SVR(kernel=self.kernel, C=self.C)
            elif self.kernel == 'poly':
                self.model = SVR(kernel=self.kernel,
                                 degree=self.degree, gamma=self.gamma, C=self.C)
            else:
                self.model = SVR(kernel=self.kernel,
                                 gamma=self.gamma, C=self.C)
            self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        """
        Parameters
        ----------
        X_test : A matrix or dataframe of test input variable values to compute predictions.

        Returns
        -------
        A numeric array for predictions at the data points in X_test.

        """

        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        if len(self.X_train.shape) > 1:
            if X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    "The number of features in train and test set must be same.")

        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
        X_test = self.scale_features.transform(X_test)

        predictions = self.model.predict(X_test)
        if not self.is_discrete:
            predictions = self.scale_target.inverse_transform(
                predictions.reshape(-1, 1)).squeeze()

        return predictions
