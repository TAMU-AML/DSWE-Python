# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_features, validate_inputs
from ._GPMethods import *
from ._tempGP_subroutine import *


class TempGP(object):

    """
    The temporal overfitting problem with applications in wind power curve modeling.
    A Gaussian process based power curve model which explicitly models the temporal aspect of the power curve. 
    The model consists of two parts: f(x) and g(t).

    References
    ----------
    Prakash, Tuo, and Ding, 2022, “The temporal overfitting problem with applications in wind power curve modeling,” Technometrics, accepted

    Parameters
    ----------
    opt_method: string
        Type of solver. The best working solver are ['L-BFGS-B', 'BFGS'].
        Default value is 'L-BFGS-B'.

    limit_memory: int or None
        The integer is used as sample training points during prediction to limit the total memory requirement. 
        Setting the value to None would result in no sampling, that is, full training data is used for prediction. 
        Default value is 5000.

    fast_computation: bool
        A boolean (True/False) that specifies whether to do exact inference or fast approximation.
        Default is True.

    optim_control: dict
        A dictionary of parameters passed to the Adam optimizer when fast_computation is set to True. 
        The default values have been tested rigorously and tend to strike a balance between accuracy and speed. \n
        - batch_size: Number of training points sampled at each iteration of Adam. Default value is 100.
        - learning_rate: The step size for the Adam optimizer. Default value is 0.05.
        - max_iter: The maximum number of iterations to be performed by Adam. Default value is 5000.
        - tol: Gradient tolerance. Default value is 1e-6.
        - beta1: Decay rate for the first moment of the gradient. Default value is 0.9.
        - beta2: Decay rate for the second moment of the gradient. Default value is 0.999.
        - epsilon: A small number to avoid division by zero. Default value is 1e-8.
        - logfile: A string specifying a file name to store hyperparameters value for each iteration. Default value is None.

    """

    def __init__(self, opt_method='L-BFGS-B', limit_memory=5000, fast_computation=True, optim_control={'batch_size': 100,
                                                                                                       'learning_rate': 0.05,
                                                                                                       'max_iter': 5000,
                                                                                                       'tol': 1e-6,
                                                                                                       'beta1': 0.9,
                                                                                                       'beta2': 0.999,
                                                                                                       'epsilon': 1e-8,
                                                                                                       'logfile': None}):

        if opt_method not in ['L-BFGS-B', 'BFGS']:
            raise ValueError("The opt_method must be 'L-BFGS-B' or 'BFGS'.")

        if limit_memory:
            if type(limit_memory) != int:
                raise ValueError(
                    "The limit_memory should be either an integer value or None.")

        if type(fast_computation) != type(True):
            raise ValueError(
                "The fast_computation must be either True or False.")

        if fast_computation:
            if not isinstance(optim_control, dict):
                raise ValueError(
                    "If fast_computation is True, optim_control must be a dictionary.")
            if not set(['batch_size', 'learning_rate', 'max_iter', 'tol', 'beta1', 'beta2', 'epsilon', 'logfile']) == set(list(optim_control.keys())):
                raise ValueError(
                    "optim_control must contains eight key-value pairs with ['batch_size', 'learning_rate', 'max_iter', 'tol', 'beta1', 'beta2', 'epsilon', 'logfile'] as keys.")

        self.opt_method = opt_method
        self.fast_computation = fast_computation
        self.limit_memory = limit_memory
        self.optim_control = optim_control

    def fit(self, X_train, y_train, T_train=[]):
        """Fit the TempGP from the training dataset.

        Parameters
        ----------
        X_train: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the training dataset.

        y_train: np.array
            A numeric array for response values in the training dataset.

        T_train: np.array
            A temporal array for time indices of the data points. By default, the function assigns
            natural numbers starting from 1 as the time indices.

        Returns
        -------
        TempGP
            self with trained parameters. \n
            - thinning_number: the thinning number computed by the algorithm.
            - model_F: A dictionary containing details of the model for predicting function f(x). 
                - 'X_train' is the input variable matrix for computing the cross-covariance for predictions, same as X_train unless the model is updated. See TempGP.update() method for details on updating the model.
                - 'y_train' is the response vector, again same as y_train unless the model is updated.
                - 'weighted_y' is the weighted response, that is, the response left multiplied by the inverse of the covariance matrix.
            - model_G: A dictionary containing details of the model for predicting function g(t).
                - 'residuals' is the residuals after subtracting function f(x) from the response. Used to predict g(t). See TempGP.update() method for updating the residuals.
                - 'T_train' is the time indices of the residuals, same as T_train.
            - optim_result: A dictionary containing optimized values of model f(x).
                - 'estimated_params' is estimated hyperparameters for function f(x).
                - 'obj_val' is objective value of the hyperparameter optimization for f(x).
                - 'grad_val' is gradient vector at the optimal objective value.
        """

        validate_inputs(X_train, y_train)
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if len(T_train) > 0:
            validate_inputs(X_train, T_train)
            self.T_train = np.array(T_train)
        else:
            self.T_train = np.array(list(range(1, len(self.y_train))))

        self.thinning_number = compute_thinning_number(self.X_train)
        self.databins = create_thinned_bins(
            self.X_train, self.y_train, self.thinning_number)
        self.optim_result = estimate_binned_params(
            self.databins, self.fast_computation, self.optim_control, self.opt_method)

        if self.limit_memory:
            ntrain = X_train.shape[0]
            if self.limit_memory < ntrain:
                pred_index = np.random.choice(ntrain, self.limit_memory)
                X_active = self.X_train[pred_index, :]
                y_active = self.y_train[pred_index]
            else:
                X_active = self.X_train
                y_active = self.y_train
        else:
            X_active = self.X_train
            y_active = self.y_train

        self.weighted_y = compute_weighted_y(
            X_active, y_active, self.optim_result['estimated_params'])
        self.model_F = {'X_train': X_active, 'y_train': y_active,
                        'weighted_y': self.weighted_y}

        self.residual = self.y_train - predict_GP(self.model_F['X_train'], self.model_F['weighted_y'], self.X_train,
                                                  self.optim_result['estimated_params'])

        self.model_G = {'residual': self.residual, 'time_index': self.T_train}

        return self

    def predict(self, X_test, T_test=[]):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_test: np.ndarray or pd.DataFrame
            A matrix or dataframe of test input variable values to compute predictions.

        T_test: np.array
            Temporal values of test data points.

        Returns
        -------
        np.array
            Predicted target values.

        """

        validate_features(X_test)
        X_test = np.array(X_test)

        if len(T_test) > 0:
            validate_inputs(X_test, T_test)
            T_test = np.array(T_test)

        pred_F = predict_GP(self.model_F['X_train'], self.model_F['weighted_y'], X_test,
                            self.optim_result['estimated_params'])

        if len(T_test) == 0:
            return pred_F
        else:
            pred_G = compute_local_function(
                self.model_G['residual'], self.model_G['time_index'], T_test, self.thinning_number)
            return pred_F + pred_G

    def update(self, X_update, y_update, T_update=[], replace=True, update_model_F=False):
        """Update the model when new training dataset will arrive.

        Parameters
        ----------
        X_update: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the new added dataset.

        y_update: np.array
            A numeric array for response values in the new added dataset.

        T_update: np.array
            A temporal array for time indices of the data points. By default, the function assigns
            natural numbers starting from 1 as the time indices.

        replace: bool
            A boolean to specify whether to replace the old data with the new one, or to
            add the new data while still keeping all the old data. Default is True, which
            replaces the top m rows from the old data, where m is the number of data points
            in the new data.

        update_model_F: bool
            A boolean to specify whether to update model_F as well. If the original TempGP
            model is trained on a sufficiently large dataset (say one year), updating model_F
            regularly may not result in any significant improvement, but can be computationally expensive.

        Returns
        -------
        TempGP
            self with updated trained parameter values.
        """

        validate_inputs(X_update, y_update)

        X_update = np.array(X_update)
        y_update = np.array(y_update)

        if len(T_update) > 0:
            validate_inputs(X_update, T_update)
            T_update = np.array(T_update)

        if len(T_update) > 0:
            X_update = X_update[np.argsort(T_update)]
            y_update = y_update[np.argsort(T_update)]
            T_update = T_update[np.argsort(T_update)]
        else:
            T_update = list(range(self.T_train[len(self.T_train)] + 1,
                                  self.T_train[len(self.T_train)] + len(y_update)))

        if replace:
            if(len(y_update) < len(self.y_train)):
                self.X_train = np.concatenate(
                    [self.X_train[len(y_update):], X_update])
                self.y_train = np.concatenate(
                    [self.y_train[len(y_update):], y_update])
                self.T_train = np.concatenate(
                    [self.T_train[len(y_update):], T_update])
            else:
                self.X_train = X_update
                self.y_train = y_update
                self.T_train = T_update
        else:
            self.X_train = np.concatenate([self.X_train, X_update])
            self.y_train = np.concatenate([self.y_train, y_update])
            self.T_train = np.concatenate([self.T_train, T_update])

        if update_model_F:
            self.model_F['X_train'] = self.X_train
            self.model_F['y_train'] = self.y_train
            self.weighted_y = compute_weighted_y(
                self.model_F['X_train'], self.model_F['y_train'], self.optim_result['estimated_params'])
            self.model_F['weighted_y'] = self.weighted_y
            self.residual = self.y_train - predict_GP(
                self.model_F['X_train'], self.model_F['weighted_y'], self.X_train, self.optim_result['estimated_params'])
            self.model_G['residual'] = self.residual
        else:
            update_residual = y_update - \
                predict_GP(self.model_F['X_train'], self.model_F['weighted_y'], X_update,
                           self.optim_result['estimated_params'])

            if replace:
                if len(y_update) < len(self.y_train):
                    self.model_G['residual'] = np.concatenate(
                        [self.model_G['residual'][len(y_update):], update_residual])
                else:
                    self.model_G['residual'] = update_residual
            else:
                self.model_G['residual'] = np.concatenate(
                    [self.model_G['residual'], update_residual])

        self.model_G['time_index'] = self.T_train

        return self
