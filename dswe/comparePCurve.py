# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ._comparePCurve_subroutine import *
from .funGP import *
from .covmatch import *


class ComparePCurve(object):

    """
    References
    ----------
    Ding, Kumar, Prakash, Kio, Liu, Liu, and Li, 2021, “A case study of space-time performance comparison of wind turbines on a wind farm,” Renewable Energy, Vol. 171, pp. 735-746.

    Parameters
    ----------
    Xlist: list
        A list, consisting of data sets to match, also each of the individual data set can be 
        a matrix with each column corresponding to one input variable.

    ylist: list
        A list, consisting of data sets to match, and each list is an array that corresponds to target 
        values of the data sets.

    testcol: list
        A list stating column number of covariates to used in generating test set. 
        Maximum of two columns to be used.

    testset: np.array
        Test points at which the functions will be compared.
        Default is set to None, means calculate at the runtime.

    circ_pos: list or int
        A list or array stating the column position of circular variables.
        An integer when only one circular variable present. 
        Default value is None.

    thresh: float or list
        A numerical or a list of threshold values for each covariates, against which matching happens.
        It should be a single value or a list of values representing threshold for each of the covariate.
        Default value is 0.2.

    conf_level: float
        A single value representing the statistical significance level for constructing the band.
        Default value is 0.95.

    grid_size: list
        A list or numpy array to be used in constructing test set, should be provided when
        testset is None, else it is ignored. Default is [50,50] for 2-dim input which
        is converted internally to a default of [1000] for 1-dim input. Total number of
        test points (product of grid_size elements components) must be less than or equal
        to 2500.

    power_bins: int
        A integer stating the number of power bins for computing the scaled difference.
        Default value is 15.

    bseline: int
        An integer between 0 to 2, where 1 indicates to use power curve of first dataset
        as the base for metric calculation, 2 indicates to use the power curve of second
        dataset as the base, and 0 indicates to use the average of both power curves as
        the base. Default is set to 1.

    limit_memory: bool
        A boolean (True/False) indicating whether to limit the memory use or not. 
        Default is True. If set to True, 5000 datapoints are randomly sampled 
        from each dataset under comparison for inference.  

    opt_method: string
        A string specifying the optimization method to be used for hyperparameter 
        estimation. The best working solver are ['L-BFGS-B', 'BFGS'].
        Default is set to 'L-BFGS-B'.

    sample_size: dict
        A dictionary with two keys: optim_size and band_size, denoting the sample size for each dataset for 
        hyperparameter optimization and confidence band computation, respectively, when limit_memory = TRUE. 
        Default value is list(optim_size = 500,band_size = 5000).

    rng_seed: int
        Random number genrator (rng) seed for sampling data when limit_memory = TRUE. Default value is 1.  

    Returns
    -------
    ComparePCurve
        self with trained parameters. \n
        - weighted_diff: a numeric, % difference between the functions weighted using the density of the covariates.
        - weighted_stat_diff: a numeric, % statistically significant difference between the functions weighted using the density of the covariates.
        - scaled_diff: a numeric, % difference between the functions scaled to the orginal data. 
        - scaled_stat_diff: a numeric, % statistically significant difference between the functions scaled to the orginal data.
        - unweighted_diff: a numeric, % difference between the functions unweighted.
        - unweighted_stat_diff: a numeric, % statistically significant difference between the functions unweighted.
        - reduction_ratio: a list consisting of shrinkage ratio of features used in testset.
        - mu1: An array of test prediction for first data set.
        - mu2: An array of test prediction for second data set.
        - mu_diff: An array of pointwise difference between the predictions from the two datasets (mu2-mu1).
        - band: An array of the allowed statistical difference between functions at testpoints in testset.
        - conf_level: A numeric representing the statistical significance level for constructing the band.
        - estimated_params: A list of estimated hyperparameters for GP.
        - testset: an array/matrix of the test points either provided by user, or generated internally.
        - matched_data_X: a list of features of two matched datasets as generated by covariate matching.
        - matched_data_y: a list of target of two matched datasets as generated by covariate matching.
    """

    def __init__(self, Xlist, ylist, testcol, testset=None, circ_pos=None, thresh=0.2, conf_level=0.95, grid_size=[50, 50],
                 power_bins=15, baseline=1, limit_memory=True, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, rng_seed=1):

        validate_matching(Xlist, ylist)

        if not (isinstance(testcol, list) or isinstance(testcol, np.ndarray)):
            raise ValueError(
                "The testcol must be provided in a list or 1d-array.")
        if len(testcol) > 2:
            raise ValueError("Maximum two columns to be used.")

        if testset is None:
            if not (isinstance(grid_size, list) or isinstance(grid_size, np.ndarray)):
                raise ValueError(
                    "The grid_size must be provided in a list or 1d-array.")
            elif len(grid_size) != 2 and len(testcol) == 2:
                raise ValueError(
                    "The length of grid_size should be equal to two when length of testcol is equal to two.")

            if len(testcol) == 1 and len(grid_size) == 2 and grid_size == [50, 50]:
                # Convert the 2-dim default grid_size to 1-dim default internally when len(testcol) == 1.
                grid_size = [1000]
            elif len(testcol) == 1 and len(grid_size) != 1:
                raise ValueError(
                    "The length of grid_size should be equal to one when length of testcol is equal to one, or use the default grid_size option.")

            if np.prod(grid_size) > 2500:
                raise ValueError(
                    "The number of test points should be less than or equal to 2500; reduce grid_size. The total number of test points are product of values in grid_size.")

        if circ_pos:
            if not (isinstance(circ_pos, list) or isinstance(circ_pos, np.ndarray) or type(circ_pos) == int):
                raise ValueError(
                    "The circ_pos should be a list or 1d-array or single integer value or set to None.")
            if type(circ_pos) == int:
                circ_pos = [circ_pos]

        if isinstance(thresh, list) or isinstance(thresh, np.ndarray):
            if len(thresh) > 0:
                if len(thresh) != Xlist[0].shape[1]:
                    raise ValueError(
                        "The thresh must be a single value, or list or 1d array with weight for each covariate.")

        if type(conf_level) != int and type(conf_level) != float or conf_level < 0 or conf_level > 1:
            raise ValueError(
                "The conf_level be a numeric value between 0 and 1")

        if baseline not in [0, 1, 2]:
            raise ValueError("The basline must be an integer between 0 and 2.")

        if type(limit_memory) != type(True):
            raise ValueError("The limit_memory must be either True or False.")

        if limit_memory:
            if not isinstance(sample_size, dict):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")
            if not set(['optim_size', 'band_size']) == set(list(sample_size.keys())):
                raise ValueError(
                    "If limitMemory is True, sample_size must be a dictionary with two named items: optim_size and band_size.")

        if type(rng_seed) != int:
            raise ValueError("The range seed must be a single integer value.")

        if opt_method not in ['L-BFGS-B', 'BFGS']:
            raise ValueError("The opt_method must be 'L-BFGS-B' or 'BFGS'.")

        self.conf_level = conf_level
        self.opt_method = opt_method
        self.rng_seed = rng_seed

        self.Xlist = Xlist
        self.ylist = ylist

        for i in range(2):
            self.Xlist[i] = np.array(self.Xlist[i])
            self.ylist[i] = np.array(self.ylist[i]).reshape(-1, 1)

        for i in range(2):
            if len(self.Xlist[i].shape) == 1:
                self.Xlist[i] = self.Xlist[i].reshape(-1, 1)

        if self.Xlist[0].shape[1] != self.Xlist[1].shape[1]:
            raise ValueError(
                "The number of columns in both the dataset should be the same.")

        self.testset = testset
        if self.testset is not None:
            if len(self.testset.shape) == 1:
                self.testset = self.testset.reshape(-1, 1)

            if self.Xlist[0].shape[1] != self.testset.shape[1]:
                raise ValueError(
                    "The number of columns in input and testset should be same.")

        if self.testset is None:
            if self.Xlist[0].shape[1] != len(testcol):
                raise ValueError(
                    "The length of testcol should match the number of input columns.")

        result_matching = CovMatch(self.Xlist, self.ylist, circ_pos, thresh)
        self.matched_data_X = result_matching.matched_data_X
        self.matched_data_y = result_matching.matched_data_y

        if self.testset is None:
            self.testset = generate_test_set(
                self.matched_data_X, testcol, grid_size)

        _mdata_X = [self.matched_data_X[0][:, testcol],
                    self.matched_data_X[1][:, testcol]]
        _mdata_y = [self.matched_data_y[0], self.matched_data_y[1]]
        result_GP = FunGP(_mdata_X, _mdata_y, self.testset, self.conf_level,
                          limit_memory, self.opt_method, sample_size=sample_size, rng_seed=self.rng_seed)

        self.mu1 = result_GP.mu1
        self.mu2 = result_GP.mu2
        self.mu_diff = result_GP.mu_diff
        self.band = result_GP.band
        self.estimated_params = result_GP.params

        self.weighted_diff = compute_weighted_diff(
            self.Xlist, self.mu1, self.mu2, self.testset, testcol, baseline)

        self.weighted_stat_diff = compute_weighted_stat_diff(
            self.Xlist, self.mu1, self.mu2, self.band, self.testset, testcol, baseline)

        self.scaled_diff = compute_scaled_diff(
            self.ylist, self.mu1, self.mu2, power_bins, baseline)

        self.scaled_stat_diff = compute_scaled_stat_diff(
            self.ylist, self.mu1, self.mu2, self.band, power_bins, baseline)

        self.unweighted_diff = compute_diff(self.mu1, self.mu2, baseline)

        self.unweighted_stat_diff = compute_stat_diff(
            self.mu1, self.mu2, self.band, baseline)

        self.reduction_ratio = compute_ratio(
            self.Xlist, self.matched_data_X, testcol)

    def compute_weighted_difference(self, weights, baseline=1, stat_diff=False):
        """
        Computes percentage weighted difference between power curves based on user provided weights
        instead of the weights computed from the data.

        Parameters
        ----------
        weights: list
            a list of user specified weights for each element of mu_diff. It can be based
            on any probability distribution of user choice. The weights must sum to 1.

        baseline: int
            An integer between 1 to 2, where 1 indicates to use mu1 predictions from the power curve and 
            2 indicates to use mu2 predictions from the power curve as obtained from ComparePCurve() function. 
            The mu1 and mu2 corresponds to test prediction for first and second data set respectively.
            Default is set to 1.

        stat_diff: bool
            a boolean (True/False) specifying whether to compute the statistical significant difference or not.
            Default is set to False, i.e. statistical significant difference is not computed.
            If set to true, band generated from ComparePCurve() function to be used.

        Returns
        -------
        float
            numeric percentage weighted difference or statistical significant percetage weighted difference
            based on whether statDiff is set to False or True.
        """

        weights = np.array(weights)

        if not (isinstance(weights, list) or isinstance(weights, np.ndarray)):
            raise ValueError("The weights must be a numeric vector.")

        if len(weights) != len(self.mu_diff):
            raise ValueError(
                "The length of weights vector must be equal to the length of mu_diff which has calculated in creating this class instance")
        if abs(weights.sum() - 1) >= 1e-3:
            raise ValueError(
                "The weights must sume to 1 for being a valid weights.")

        if type(stat_diff) != type(True):
            raise ValueError("The stat_diff must be either True or False.")

        if baseline == 1:
            return compute_weighted_diff_extern(self.mu_diff, weights, self.mu1, stat_diff, self.band)
        else:
            return compute_weighted_diff_extern(self.mu_diff, weights, self.mu2, stat_diff, self.band)
