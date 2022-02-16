# Copyright (c) 2022 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_matching
from ._covmatch_subroutine import *


class CovMatch(object):

    """
    Covariate Matching
    ------------------
    The class aims to take list of two data sets and returns the after matched data sets using user
    specified covariates and threshold.

    References
    ----------
    Ding, Y. (2019). Data Science for Wind Energy. Chapman & Hall, Boca Raton, FL.

    Parameters
    ----------
    data : A list, consisting of data sets to match, also each of the individual data set can be 
           a matrix with each column corresponding to one input variable. 

    circ_pos : A list or array stating the column position of circular variables.

    thresh: A numerical or a list of threshold values for each covariates, against which matching happens.
            It should be a single value or a list of values representing threshold for each of the covariate.

    priority: A boolean, default value False, otherwise computes the sequence of matching.

    Returns
    -------
    A matched object (dictionary) of class CovMatch.

        original_data: The data sets provided for matching.
        matched_data: The data sets after matching.
        min_max_original: The minimum and maximum value in original data for each covariates used in matching.
        min_max_matched: The minimum and maximum value in matched data for each covariates used in matching.

    """

    def __init__(self, data, circ_pos=None, thresh=0.1, priority=False):

        validate_matching(data, circ_pos, thresh)

        self.data = data
        for i in range(2):
            self.data[0] = np.array(self.data[0])
            self.data[1] = np.array(self.data[1])

        if priority:
            idx = np.argsort(-(np.abs(np.mean(self.data[0],
                                              axis=0) - np.mean(self.data[1], axis=0))))
            self.data[0] = self.data[0][:, idx]
            self.data[1] = self.data[1][:, idx]

        self.thresh = thresh
        self.circ_pos = circ_pos

        datalist = [[self.data[0], self.data[1]],
                    [self.data[1], self.data[0]]]
        matched_asym = [[]] * 2

        for i in range(2):
            matched_asym[i] = matching(
                datalist[i], self.thresh, self.circ_pos)

        matched1 = [np.squeeze(matched_asym[0][1]),
                    np.squeeze(matched_asym[0][0])]
        matched2 = [np.squeeze(matched_asym[1][0]),
                    np.squeeze(matched_asym[1][1])]

        self.matched_data = [[]] * 2

        self.matched_data[0] = np.unique(np.concatenate(
            (matched1[1], matched2[1])), axis=0)
        self.matched_data[1] = np.unique(np.concatenate(
            (matched1[0], matched2[0])), axis=0)

        self.min_max_original = min_max(self.data)
        self.min_max_matched = min_max(self.matched_data)
