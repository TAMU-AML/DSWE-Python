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
