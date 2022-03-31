import numpy as np
import pandas as pd
import json

from dswe import FunGP
from dswe import CovMatch
from dswe import generate_test_set, compute_weighted_diff

vg = pd.read_csv(
    '../../Downloads/DS/Turbine_Upgrade_Dataset/Turbine Upgrade Dataset(VG Pair).csv')

xcol = [3, 4, 5, 6]
ycol = 10
circ_pos = [1]
testcol = [0, 1]
grid_size = [50, 50]

Xlist = [vg[vg['upgrade status'] == 0].to_numpy()[:, xcol].astype(
    float), vg[vg['upgrade status'] == 1].to_numpy()[:, xcol].astype(float)]
ylist = [vg[vg['upgrade status'] == 0].to_numpy()[:, ycol].astype(
    float), vg[vg['upgrade status'] == 1].to_numpy()[:, ycol].astype(float)]

result_matching = CovMatch(Xlist, ylist, circ_pos)
matched_data_X = result_matching.matched_data_X
matched_data_y = result_matching.matched_data_y
# result_matching = json.load(open('../../Downloads/DSWE-Package/testing/matchedData.json'))
# result_matching[0] = pd.DataFrame(result_matching[0])
# result_matching[1] = pd.DataFrame(result_matching[1])
# matched_data_X = [result_matching[0].iloc[:,xcol].values, result_matching[1].iloc[:,xcol].values]
# matched_data_y = [result_matching[0].iloc[:,10].values, result_matching[1].iloc[:,10].values]

testset = generate_test_set(matched_data_X, testcol, grid_size)
# testset = pd.read_csv('../../Downloads/DSWE-Package/testing/testset.csv')
# testset =  testset.values

optim_idx = json.load(
    open('../../Downloads/DSWE-Package/testing/optimIdx.json'))
band_idx = json.load(open('../../Downloads/DSWE-Package/testing/bandIdx.json'))
estimated_params = json.load(
    open('../../Downloads/DSWE-Package/testing/estimatedParams.json'))

# to start index with 0
for i in range(len(optim_idx)):
    if optim_idx[i] is not None:
        for j in range(len(optim_idx[i])):
            optim_idx[i][j] = optim_idx[i][j] - 1
for i in range(len(band_idx)):
    if band_idx[i] is not None:
        for j in range(len(band_idx[i])):
            band_idx[i][j] = band_idx[i][j] - 1

_mdata_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
_mdata_y = [matched_data_y[0], matched_data_y[1]]
result_GP = FunGP(_mdata_X, _mdata_y, testset, conf_level=0.95,
                  limit_memory=True, opt_method='L-BFGS-B', sample_size={'optim_size': 500, 'band_size': 5000}, rng_seed=1, optim_idx=optim_idx, band_idx=band_idx, params=estimated_params)

print("Weighted Diff: ", compute_weighted_diff(
    Xlist, result_GP.mu1, result_GP.mu2, testset, testcol, baseline=1))
