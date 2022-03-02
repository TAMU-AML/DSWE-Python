from matplotlib.pyplot import angle_spectrum
import numpy as np
from _comparePCurve_subroutine import *
from funGP import *
from covmatch import *


class ComparePCurve(object):

    def __init__(self, Xlist, ylist, testcol, testX=None, circ_pos=None, thresh=0.1, conf_level=0.95, grid_size=[20, 20],
                 power_bins=15, baseline=2, limit_memory=False, opt_method='L-BFGS-B',
                 sample_size={'optim_size': 500, 'band_size': 5000}, range_seed=1):

        self.Xlist = Xlist
        self.ylist = ylist
        self.Xlist[0] = np.array(self.Xlist[0])
        self.Xlist[1] = np.array(self.Xlist[1])
        self.ylist[0] = np.array(self.ylist[0]).reshape(-1, 1)
        self.ylist[1] = np.array(self.ylist[1]).reshape(-1, 1)

        self.testX = testX

        result_matching = CovMatch(self.Xlist, self.ylist, circ_pos, thresh)

        if not self.testX:
            self.testX = generate_test_set(
                result_matching.matched_data_X, testcol, grid_size)

        np.random.seed(range_seed)
        _mdata = [result_matching.matched_data_X[0][:, testcol],
                  result_matching.matched_data_X[1][:, testcol]]
        result_GP = FunGP(_mdata, result_matching.matched_data_y,
                          self.testX, conf_level, limit_memory, opt_method)

        self.weight_diff = compute_weighted_diff(
            self.Xlist, result_GP.mu1, result_GP.mu2, self.testX, testcol, baseline)

        self.weight_stat_diff = compute_weighted_stat_diff(
            self.Xlist, result_GP.mu1, result_GP.mu2, result_GP.band, self.testX, testcol, baseline)

        self.scale_diff = compute_scaled_diff(
            self.ylist, result_GP.mu1, result_GP.mu2, power_bins, baseline)

        self.scale_stat_diff = compute_scaled_stat_diff(
            self.ylist, result_GP.mu1, result_GP.mu2, result_GP.band, power_bins, baseline)

        self.unweight_diff = compute_diff(
            result_GP.mu1, result_GP.mu2, baseline)

        self.unweight_stat_diff = compute_stat_diff(
            result_GP.mu1, result_GP.mu2, result_GP.band, baseline)

        self.reduction_ratio = compute_ratio(
            self.Xlist, result_matching.matched_data_X, testcol)
