# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd


def validate_inputs(X, y):
    """Validates the inputs to model."""

    if not (isinstance(X, list) or isinstance(pd.DataFrame(X), pd.DataFrame) or isinstance(np.array(X), np.ndarray)):
        raise ValueError(
            "The features data should be either of list or numpy array or dataframe.")

    if not (isinstance(y, list) or isinstance(pd.DataFrame(y), pd.DataFrame) or isinstance(np.array(y), np.ndarray)):
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

    if not (isinstance(X, list) or isinstance(pd.DataFrame(X), pd.DataFrame) or isinstance(np.array(X), np.ndarray)):
        raise ValueError(
            "The features data should be either of list or numpy array or dataframe.")

    if np.isnan(np.array(X)).any():
        raise ValueError("The data should not contains any null value.")

    if np.isinf(np.array(X)).any():
        raise ValueError("The data must not have any infinity value.")

    if not np.issubdtype(np.array(X).dtype, np.number):
        raise ValueError("The data should have only numeric values.")


def validate_matching(data, circ_pos, thresh):

    if not isinstance(data, list):
        raise ValueError("Data must be a list containing data sets.")

    if len(data) != 2:
        raise ValueError(
            "The number of data sets to match should be equal to two.")

    if isinstance(data, list):
        if not (isinstance(data[0], list) or isinstance(pd.DataFrame(data[0]), pd.DataFrame) or isinstance(np.array(data[0]), np.ndarray)):
            raise ValueError(
                "The features of first dataset should be either of list or numpy array or dataframe.")
        if not (isinstance(data[1], list) or isinstance(pd.DataFrame(data[1]), pd.DataFrame) or isinstance(np.array(data[1]), np.ndarray)):
            raise ValueError(
                "The features of second dataset should be either of list or numpy array or dataframe.")

    if circ_pos:
        if not (isinstance(circ_pos, list) or isinstance(np.array(circ_pos), np.ndarray)):
            raise ValueError(
                "The circular position must be provided in a list or 1d-array.")

    if (isinstance(thresh, list) or isinstance(np.array(thresh), np.ndarray)):
        if len(thresh) > 0:
            if len(thresh) != data[0].shape[1]:
                raise ValueError(
                    "The thresh must be a single value, or list or 1d array with weight for each covariate.")
