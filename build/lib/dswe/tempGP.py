# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from .utils import validate_inputs


class TempGP(object):

    def __init__(self):
        return self

    def fit(self, X, y):

        validate_inputs(X, y)

        self.X = X
        self.y = y
