# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_features, validate_inputs
from ._GPMethods import *
from ._tempGP_subroutine import *


class TempGP(object):

    """
    Temporal Gaussian Process (TempGP)
    ---------------------------------
    Paper: The temporal overfitting problem with applications in wind power curve modeling.
    Description: A Gaussian process based power curve model which explicitly models 
                the temporal aspect of the power curve. 
                The model consists of two parts: f(x) and g(t).

    Parameters
    ----------
    opt_method: Type of solver. The best working solver are ['L-BFGS-B', 'BFGS'].

    """

    def __init__(self, opt_method='L-BFGS-B'):
        self.opt_method = opt_method

    def fit(self, X, y, T=[]):
        """Fit the TempGP from the training dataset.

        Parameters
        ----------
        X : A matrix with each column corresponding to one input variable. \
            {array-like, sparse matrix} of shape (n_samples, n_features). \


        y : A vector with each element corresponding to the output at the corresponding
        row of X.
        {array-like, sparse matrix} of shape (n_samples,)
        Target values.

        T : A vector for time indices of the data points. By default, the function assigns
            natural numbers starting from 1 as the time indices.
            {array-like, sparse matrix} of shape (n_samples,)
            Temporal values.

        Returns
        -------
        A fitted object of class TempGP.

        thinning_number : the thinning number computed by the algorithm.
        model_F :  A dictionary containing details of the model for predicting function f(x).
            -   X : The input variable matrix for computing the cross-covariance for predictions, same
                    as X unless the model is updated. See TempGP.update() method for details on
                    updating the model.
            -   y : The response vector, again same as y unless the model is updated.
            -   weighted_y : The weighted response, that is, the response left multiplied by the inverse of
                             the covariance matrix.
        model_G: A dictionary containing details of the model for predicting function g(t).
            -   residuals : The residuals after subtracting function f(x) from the response. Used to
                            predict g(t). See TempGP.update() method for updating the residuals.
            -   T : The time indices of the residuals, same as T.

        optim_result : A dictionary containing optimized values of model f(x).
            -   estimated_params :  estimated hyperparameters for function f(x).
            -   obj_val : objective value of the hyperparameter optimization for f(x).
            -   grad_val : gradient vector at the optimal objective value.

        References
        ----------
        Prakash, A., Tuo, R., & Ding, Y. (2020). "The temporal overfitting problem with applications
        in wind power curve modeling." arXiv preprint arXiv:2012.01349. <https://arxiv.org/abs/
        2012.01349>.
        """

        validate_inputs(X, y)
        self.X = np.array(X)
        self.y = np.array(y)

        if len(T) > 0:
            validate_inputs(X, T)
            self.T = np.array(T)
        else:
            self.T = np.array(list(range(1, len(self.y))))

        self.thinning_number = compute_thinning_number(self.X)
        self.databins = create_thinned_bins(
            self.X, self.y, self.thinning_number)
        self.optim_result = estimate_binned_params(
            self.databins, self.opt_method)
        self.weighted_y = compute_weighted_y(
            self.X, self.y, self.optim_result['estimated_params'])
        self.residual = self.y - predict_GP(self.X, self.weighted_y, self.X,
                                            self.optim_result['estimated_params'])

        self.model_F = {'X': self.X, 'y': self.y,
                        'weighted_y': self.weighted_y}
        self.model_G = {'residual': self.residual, 'time_index': self.T}

        return self

    def predict(self, X, T=[]):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : A matrix with each column corresponding to one input variable.
            array-like of shape (n_samples, n_features)
            Test samples.

        T : A vector for time indices of the data points.
            {array-like, sparse matrix} of shape (n_samples,)
            Temporal values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
           Predicted target values.

        """

        validate_features(X)
        X = np.array(X)

        if len(T) > 0:
            validate_inputs(X, T)
            T = np.array(T)

        pred_F = predict_GP(self.model_F['X'], self.model_F['weighted_y'], X,
                            self.optim_result['estimated_params'])

        if len(T) == 0:
            return pred_F
        else:
            pred_G = compute_local_function(
                self.model_G['residual'], self.model_G['time_index'], T, self.thinning_number)
            return pred_F + pred_G

    def update(self, X, y, T=[], replace=True, update_model_F=False):
        """Update the model when new training dataset will arrive.

        Parameters
        ----------
        X : A matrix with each column corresponding to one input variable.
            {array-like, sparse matrix} of shape (n_samples, n_features)
            New training data.

        y : A vector with each element corresponding to the output at the corresponding
            row of X.
            {array-like, sparse matrix} of shape (n_samples,)
            New target values.

        T : A vector with time indices of the new datapoints. If None, the function assigns
            natural numbers starting with one larger than the existing time indices in existing T.
            {array-like, sparse matrix} of shape (n_samples,)
            New temporal values.

        replace : A boolean to specify whether to replace the old data with the new one, or to
                add the new data while still keeping all the old data. Default is True, which
                replaces the top m rows from the old data, where m is the number of data points
                in the new data.

        update_model_F : A boolean to specify whether to update model_F as well. If the original TempGP
                        model is trained on a sufficiently large dataset (say one year), updating model_F
                        regularly may not result in any significant improvement, but can be computationally expensive.

        Returns
        -------
        An updated object of class TempGP.
        """

        validate_inputs(X, y)

        X = np.array(X)
        y = np.array(y)

        if len(T) > 0:
            validate_inputs(X, T)
            T = np.array(T)

        if len(T) > 0:
            X = X[np.argsort(T)]
            y = y[np.argsort(T)]
            T = T[np.argsort(T)]
        else:
            T = list(range(self.T[len(self.T)] + 1,
                     self.T[len(self.T)] + len(y)))

        if replace:
            if(len(y) < len(self.y)):
                self.X = np.concatenate([self.X[len(y):], X])
                self.y = np.concatenate([self.y[len(y):], y])
                self.T = np.concatenate([self.T[len(y):], T])
            else:
                self.X = X
                self.y = y
                self.T = T
        else:
            self.X = np.concatenate([self.X, X])
            self.y = np.concatenate([self.y, y])
            self.T = np.concatenate([self.T, T])

        if update_model_F:
            self.model_F['X'] = self.X
            self.model_F['y'] = self.y
            self.weighted_y = compute_weighted_y(
                self.model_F['X'], self.model_F['y'], self.optim_result['estimated_params'])
            self.model_F['weighted_y'] = self.weighted_y
            self.residual = self.y - predict_GP(
                self.model_F['X'], self.model_F['weighted_y'], self.X, self.optim_result['estimated_params'])
            self.model_G['residual'] = self.residual
        else:
            update_residual = y - \
                predict_GP(self.model_F['X'], self.model_F['weighted_y'], X,
                           self.optim_result['estimated_params'])

            if replace:
                if len(y) < len(self.y):
                    self.model_G['residual'] = np.concatenate(
                        [self.model_G['residual'][len(y):], update_residual])
                else:
                    self.model_G['residual'] = update_residual
            else:
                self.model_G['residual'] = np.concatenate(
                    [self.model_G['residual'], update_residual])

        self.model_G['time_index'] = self.T

        return self
