# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd


def validate_inputs(X, y):
    """Validates the inputs to model."""

    if not (isinstance(X, list) or isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
        raise ValueError(
            "The features data should be either of list or numpy array or dataframe.")

    if not (isinstance(y, list) or isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray)):
        raise ValueError(
            "The target data should be either of list or numpy array or dataframe.")

    if len(X) != len(y):
        raise ValueError(
            "The features and targets should have same number of data points.")

    if np.isnan(np.array(X)).any() or np.isnan(np.array(y)).any():
        raise ValueError("The data should not contains any null value.")

    if np.isinf(np.array(X)).any() or np.isinf(np.array(y)).any():
        raise ValueError("The data must not have any infinity value.")

    if not (np.issubdtype(np.array(X).dtype, np.number) or np.issubdtype(np.array(y).dtype, np.number)):
        raise ValueError("The data should have only numeric values.")


def validate_features(X):
    """Validates the inputs without target to model."""

    if not (isinstance(X, list) or isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
        raise ValueError(
            "The features data should be either of list or numpy array or dataframe.")

    if np.isnan(np.array(X)).any():
        raise ValueError("The data should not contains any null value.")

    if np.isinf(np.array(X)).any():
        raise ValueError("The data must not have any infinity value.")

    if not np.issubdtype(np.array(X).dtype, np.number):
        raise ValueError("The data should have only numeric values.")


def validate_matching(Xlist, ylist):

    if not (isinstance(Xlist, list) or isinstance(Xlist, np.ndarray)):
        raise ValueError(
            "The Xlist must be a list containing the data sets.")

    if len(Xlist) != 2:
        raise ValueError(
            "The number of data sets to match should be equal to two.")

    if ylist:
        if len(Xlist) != len(ylist):
            raise ValueError(
                "The length of Xlist and ylist must be same and equal to two.")

    if not (isinstance(Xlist[0], list) or isinstance(Xlist[0], pd.DataFrame) or isinstance(Xlist[0], np.ndarray)):
        raise ValueError(
            "The features of first dataset should be either of list or numpy array or dataframe.")
    if not (isinstance(Xlist[1], list) or isinstance(Xlist[1], pd.DataFrame) or isinstance(Xlist[1], np.ndarray)):
        raise ValueError(
            "The features of second dataset should be either of list or numpy array or dataframe.")

    if ylist:
        if not (isinstance(ylist[0], list) or isinstance(ylist[0], pd.DataFrame) or isinstance(ylist[0], np.ndarray)):
            raise ValueError(
                "The target value of first dataset should be either of list or numpy array or dataframe.")
        if not (isinstance(ylist[1], list) or isinstance(ylist[1], pd.DataFrame) or isinstance(ylist[1], np.ndarray)):
            raise ValueError(
                "The target value of second dataset should be either of list or numpy array or dataframe.")

    if ylist:
        if len(Xlist[0]) != len(ylist[0]):
            raise ValueError(
                "The features and targets values of first dataset should have same number of data points.")
        if len(Xlist[1]) != len(ylist[1]):
            raise ValueError(
                "The features and targets values of second dataset should have same number of data points.")

    if np.isnan(np.array(Xlist[0])).any() or np.isnan(np.array(Xlist[1])).any():
        raise ValueError(
            "The features data should not contains any null value.")
    if np.isinf(np.array(Xlist[0])).any() or np.isinf(np.array(Xlist[1])).any():
        raise ValueError("The features data must not have any infinity value.")

    if ylist:
        if np.isnan(np.array(ylist[0])).any() or np.isnan(np.array(ylist[1])).any():
            raise ValueError(
                "The target data should not contains any null value.")
        if np.isinf(np.array(ylist[0])).any() or np.isinf(np.array(ylist[1])).any():
            raise ValueError(
                "The target data must not have any infinity value.")
