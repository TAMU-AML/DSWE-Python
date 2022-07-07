# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_matching
from ._covmatch_subroutine import *


class CovMatch(object):

    """
    Parameters
    ----------
    Xlist: list
        A list, consisting of data sets to match, also each of the individual data set can be 
        a matrix with each column corresponding to one input variable.

    ylist: list
        A list, consisting of data sets to match, and each list is a array corresponds to target 
        values of the data sets.

    circ_pos: list or int
        A list or array stating the column position of circular variables.
        An integer when only one circular variable present. Default is set to None.

    thresh: float or list
        A numerical or a list of threshold values for each covariates, against which matching happens.
        It should be a single value or a list of values representing threshold for each of the covariate.
        Default value is 0.2.

    priority: bool
        A boolean, default value False, otherwise computes the sequence of matching.
        Default is False.

    Returns
    -------
    CovMatch
        self with trained parameters. \n
        - matched_data_X: The variable values of datasets after matching.
        - matched_data_y: The response values of datasets after matching (if provided, otherwise None).
        - min_max_original: The minimum and maximum value in original data for each covariates used in matching.
        - min_max_matched: The minimum and maximum value in matched data for each covariates used in matching.

    """

    def __init__(self, Xlist, ylist=None, circ_pos=None, thresh=0.2, priority=False):

        validate_matching(Xlist, ylist)

        if circ_pos:
            if not (isinstance(circ_pos, list) or isinstance(circ_pos, np.ndarray) or type(circ_pos) == int):
                raise ValueError(
                    "The circ_pos should be a list or 1d-array or single integer value or set to None.")
            if type(circ_pos) == int:
                circ_pos = [circ_pos]

        if (isinstance(thresh, list) or isinstance(thresh, np.ndarray)):
            if len(thresh) > 0:
                if len(thresh) != Xlist[0].shape[1]:
                    raise ValueError(
                        "The thresh must be a single value, or list or 1d array with weight for each covariate.")

        if type(priority) != type(True):
            raise ValueError("The priority must be either True or False.")

        self.Xlist = Xlist
        self.ylist = ylist
        self.Xlist[0] = np.array(self.Xlist[0])
        self.Xlist[1] = np.array(self.Xlist[1])

        if self.ylist:
            self.ylist[0] = np.array(self.ylist[0]).reshape(-1, 1)
            self.ylist[1] = np.array(self.ylist[1]).reshape(-1, 1)

        if priority:
            idx = np.argsort(
                -(np.abs(np.mean(self.Xlist[0], axis=0) - np.mean(self.Xlist[1], axis=0))))
            self.Xlist[0] = self.Xlist[0][:, idx]
            self.Xlist[1] = self.Xlist[1][:, idx]

        self.thresh = thresh
        self.circ_pos = circ_pos

        datalist_X = [[self.Xlist[0], self.Xlist[1]],
                      [self.Xlist[1], self.Xlist[0]]]
        if self.ylist:
            datalist_y = [[self.ylist[0], self.ylist[1]],
                          [self.ylist[1], self.ylist[0]]]

        _matched_X = [[]] * 2

        if self.ylist:
            _matched_y = [[]] * 2

        for i in range(2):
            if self.ylist:
                _matched_X[i], _matched_y[i] = matching(
                    datalist_X[i], datalist_y[i], self.thresh, self.circ_pos)
            else:
                _matched_X[i] = matching(
                    datalist_X[i], self.ylist, self.thresh, self.circ_pos)

        matched1_X = [_matched_X[0][1], _matched_X[0][0]]
        matched2_X = [_matched_X[1][0], _matched_X[1][1]]

        if self.ylist:
            matched1_y = [_matched_y[0][1], _matched_y[0][0]]
            matched2_y = [_matched_y[1][0], _matched_y[1][1]]

        self.matched_data_X = [[]] * 2

        idx0 = np.sort(np.unique(np.concatenate(
            [matched1_X[1], matched2_X[1]]).astype(float), axis=0, return_index=True)[1])
        idx1 = np.sort(np.unique(np.concatenate(
            [matched1_X[0], matched2_X[0]]).astype(float), axis=0, return_index=True)[1])

        self.matched_data_X[0] = np.concatenate(
            [matched1_X[1], matched2_X[1]]).astype(float)[idx0]
        self.matched_data_X[1] = np.concatenate(
            [matched1_X[0], matched2_X[0]]).astype(float)[idx1]

        self.matched_data_y = None
        if self.ylist:
            self.matched_data_y = [[]] * 2
            self.matched_data_y[0] = np.squeeze(np.concatenate(
                [matched1_y[1], matched2_y[1]]).astype(float)[idx0])
            self.matched_data_y[1] = np.squeeze(np.concatenate(
                [matched1_y[0], matched2_y[0]]).astype(float)[idx1])

        self.min_max_original = min_max(self.Xlist)
        self.min_max_matched = min_max(self.matched_data_X)
