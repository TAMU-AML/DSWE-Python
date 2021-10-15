# Copyright (c) 2021 Pratyush Kumar, Abhinav Prakash, and Yu Ding

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sklearn.neighbors import KNeighborsRegressor


class KNN:

    def __init__(self, data, x_col, y_col, subset_selection=False):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.subset_selection = subset_selection

    def fit(self):
        pass

    def predict(self):
        pass

    def update(self):
        pass
