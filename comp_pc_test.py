import numpy as np
import pandas as pd
import json
from dswe import FunGP
from dswe import CovMatch
from dswe import generate_test_set, compute_weighted_diff, compute_weighted_stat_diff


# Provide dataset path, xcol, ycol, circ_pos, testcol, grid_size, limit_memory etc.
df = pd.read_csv(
    '../../Downloads/DS/Turbine_Upgrade_Dataset/Turbine Upgrade Dataset(VG Pair).csv')
xcol = [3, 4, 5, 6]
ycol = 10
circ_pos = [1]
testcol = [0, 1]
grid_size = [50, 50]
limit_memory = True
conf_level = 0.95
opt_method = 'L-BFGS-B'
sample_size = {'optim_size': 500, 'band_size': 5000}
# seed to run many iterartions to average result
rng_seed = [1, 10, 150, 343, 769, 1001, 1337, 2222, 3456, 5000]
baseline = 1


optim_idx_R = None
band_idx_R = None
estimated_params_R = None

testset_R = None
result_matching_R = None
weighted_diff_R = None


# Provide mutliple outputs which have generated by the R code. Comment if don't have.
optim_idx_R = json.load(
    open('../../Downloads/DSWE-Package/testing/optimIdx.json'))
band_idx_R = json.load(
    open('../../Downloads/DSWE-Package/testing/bandIdx.json'))
estimated_params_R = json.load(
    open('../../Downloads/DSWE-Package/testing/estimatedParams.json'))
testset_R = pd.read_csv('../../Downloads/DSWE-Package/testing/testset.csv')
result_matching_R = json.load(
    open('../../Downloads/DSWE-Package/testing/matchedData.json'))
weighted_diff_R = 5.3


# Process everything generated from R.
if optim_idx_R is not None:
    for i in range(len(optim_idx_R)):
        if optim_idx_R[i] is not None:
            optim_idx_R[i] = list(np.array(optim_idx_R[i]) - 1)

if band_idx_R is not None:
    for i in range(len(band_idx_R)):
        if band_idx_R[i] is not None:
            band_idx_R[i] = list(np.array(band_idx_R[i]) - 1)

if testset_R is not None:
    testset_R = testset_R.values

if result_matching_R is not None:
    result_matching_R[0] = pd.DataFrame(result_matching_R[0])
    result_matching_R[1] = pd.DataFrame(result_matching_R[1])
    matched_data_X_R = [result_matching_R[0].iloc[:,
                                                  xcol].values, result_matching_R[1].iloc[:, xcol].values]
    matched_data_y_R = [result_matching_R[0].iloc[:,
                                                  10].values, result_matching_R[1].iloc[:, 10].values]


Xlist = [df[df['upgrade status'] == 0].to_numpy()[:, xcol].astype(
    float), df[df['upgrade status'] == 1].to_numpy()[:, xcol].astype(float)]
ylist = [df[df['upgrade status'] == 0].to_numpy()[:, ycol].astype(
    float), df[df['upgrade status'] == 1].to_numpy()[:, ycol].astype(float)]

result_matching = CovMatch(Xlist, ylist, circ_pos)
matched_data_X = result_matching.matched_data_X
matched_data_y = result_matching.matched_data_y

testset = generate_test_set(matched_data_X, testcol, grid_size)

print("Accuracy of testset matching: {}%".format(
    np.mean(np.round(testset, 2) == np.round(testset_R, 2)) * 100))
print("Accuracy of Covmatching: {}%".format((np.mean(np.round(matched_data_X[0], 2) == np.round(
    matched_data_X_R[0], 2)) * 100 + np.mean(np.round(matched_data_X[1], 2) == np.round(matched_data_X_R[1], 2)) * 100) / 2.))


weighted_diff = {}
weighted_stat_diff = {}
order = []

# Experiment 1: Base case (Everything generated from Python)
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
wd = []
wsd = []
for rsd in rng_seed:
    result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                      sample_size=sample_size, rng_seed=rsd, optim_idx=None, band_idx=None, params=None)
    wd.append(compute_weighted_diff(Xlist, result_GP.mu1,
              result_GP.mu2, testset, testcol, baseline=baseline))
    wsd.append(compute_weighted_stat_diff(Xlist, result_GP.mu1,
               result_GP.mu2, testset, testcol, baseline=baseline))
    # saving indices and paramaters. Override after each iter. Better to set rng_seed list to one value.
    with open('ablation/optim_idx.json', 'w') as f:
        json.dump(result_GP.optim_idx, f)
    with open('ablation/band_idx.json', 'w') as f:
        json.dump(result_GP.band_idx, f)
    with open('ablation/estimated_params.json', 'w') as f:
        json.dump(result_GP.params, f)
order.append([False, False, False])
weighted_diff['experiment 1'] = wd
weighted_stat_diff['experiment 1'] = wsd
print("Experiment 1 finished.")


# Experiment 2: optim_idx=True, estimated_params=False, band_idx=False
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
wd = []
wsd = []
for rsd in rng_seed:
    result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                      sample_size=sample_size, rng_seed=rsd, optim_idx=optim_idx_R, band_idx=None, params=None)
    wd.append(compute_weighted_diff(Xlist, result_GP.mu1,
              result_GP.mu2, testset, testcol, baseline=baseline))
    wsd.append(compute_weighted_stat_diff(Xlist, result_GP.mu1,
               result_GP.mu2, testset, testcol, baseline=baseline))
order.append([True, False, False])
weighted_diff['experiment 2'] = wd
weighted_stat_diff['experiment 2'] = wsd
print("Experiment 2 finished.")


# Experiment 3: optim_idx=True, estimated_params=False, band_idx=True
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                  sample_size=sample_size, optim_idx=optim_idx_R, band_idx=band_idx_R, params=None)
order.append([True, False, True])
weighted_diff['experiment 3'] = [compute_weighted_diff(
    Xlist, result_GP.mu1, result_GP.mu2, testset, testcol, baseline=baseline)]
weighted_stat_diff['experiment 3'] = [compute_weighted_stat_diff(
    Xlist, result_GP.mu1, result_GP.mu2, testset, testcol, baseline=baseline)]
print("Experiment 3 finished.")


# Experiment 4: optim_idx=False, estimated_params=False, band_idx=True
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
wd = []
wsd = []
for rsd in rng_seed:
    result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                      sample_size=sample_size, rng_seed=rsd, optim_idx=None, band_idx=band_idx_R, params=None)
    wd.append(compute_weighted_diff(Xlist, result_GP.mu1,
              result_GP.mu2, testset, testcol, baseline=baseline))
    wsd.append(compute_weighted_stat_diff(Xlist, result_GP.mu1,
               result_GP.mu2, testset, testcol, baseline=baseline))
order.append([False, False, True])
weighted_diff['experiment 4'] = wd
weighted_stat_diff['experiment 4'] = wsd
print("Experiment 4 finished.")


# Experiment 5: optim_idx=None, estimated_params=True, band_idx=False
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
wd = []
wsd = []
for rsd in rng_seed:
    result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                      sample_size=sample_size, rng_seed=rsd, optim_idx=None, band_idx=None, params=estimated_params_R)
    wd.append(compute_weighted_diff(Xlist, result_GP.mu1,
              result_GP.mu2, testset, testcol, baseline=baseline))
    wsd.append(compute_weighted_stat_diff(Xlist, result_GP.mu1,
               result_GP.mu2, testset, testcol, baseline=baseline))
order.append([None, True, False])
weighted_diff['experiment 5'] = wd
weighted_stat_diff['experiment 5'] = wsd
print("Experiment 5 finished.")


# Experiment 6: optim_idx=None, estimated_params=True, band_idx=True
mlist_X = [matched_data_X[0][:, testcol], matched_data_X[1][:, testcol]]
mlist_y = [matched_data_y[0], matched_data_y[1]]
result_GP = FunGP(mlist_X, mlist_y, testset, conf_level=conf_level, limit_memory=limit_memory, opt_method=opt_method,
                  sample_size=sample_size, optim_idx=None, band_idx=band_idx_R, params=estimated_params_R)
order.append([None, True, True])
weighted_diff['experiment 6'] = [compute_weighted_diff(
    Xlist, result_GP.mu1, result_GP.mu2, testset, testcol, baseline=baseline)]
weighted_stat_diff['experiment 6'] = [compute_weighted_stat_diff(
    Xlist, result_GP.mu1, result_GP.mu2, testset, testcol, baseline=baseline)]

print("Experiment 6 finished.")


order = np.array(order)
result = pd.DataFrame()
result['Experiment'] = [1, 2, 3, 4, 5, 6]
result['optim_idx'] = order[:, 0]
result['estimated_params'] = order[:, 1]
result['band_idx'] = order[:, 2]
result['Weighted diff'] = weighted_diff
if weighted_diff_R:
    result['R'] = weighted_diff_R

result.to_csv('ablation/output.csv', index=False)
print("Everything completed successfully !!")
