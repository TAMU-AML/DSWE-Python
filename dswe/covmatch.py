# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .utils import validate_features, validate_inputs
from ._covmatch_subroutine import *


class CovMatch(object):

    def __init__(self, weight=0.1, circ_pos=None):

        self.weight = weight
        self.circ_pos = circ_pos

    def fit(self, data):

        self.data = data

        dname1 = [self.data[0], self.data[1]]
        dname2 = [self.data[1], self.data[0]]

        self.filelist = [dname1, dname2]
        self.matched_data = [[]] * 2

        for i in range(2):
            self.matched_data[i] = matching(
                self.filelist[i], self.weight, self.circ_pos)

        match1 = self.matched_data[0]
        matched1 = [np.squeeze(match1[1]), np.squeeze(match1[0])]

        match2 = self.matched_data[1]
        matched2 = [np.squeeze(match2[0]), np.squeeze(match2[1])]

        self.result = [[]] * 2

        self.result[0] = np.unique(np.concatenate(
            (matched1[1], matched2[1])), axis=0)
        self.result[1] = np.unique(np.concatenate(
            (matched1[0], matched2[0])), axis=0)

        self.min_max_original = min_max(self.data)
        self.min_max_matched = min_max(self.result)

        return {'original_data': self.data, 'matched_data': self.result,
                'min_max_original': self.min_max_original, 'min_max_matched': self.min_max_matched}
