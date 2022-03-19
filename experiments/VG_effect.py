import numpy as np
import pandas as pd
import pickle
from dswe.comparePCurve import ComparePCurve

data = pd.read_csv('~/workspace/Datasets/Turbine Upgrade Dataset(VG Pair).csv')

data1 = data.loc[data['upgrade status'] == 0, :]
data2 = data.loc[data['upgrade status'] == 1, :]

x_col = [3, 4, 5, 6]
y_col_test = 10
y_col_control = 11
Xlist = [data1.iloc[:, x_col], data2.iloc[:, x_col]]
ylist_test = [data1.iloc[:, [y_col_test]], data2.iloc[:, [y_col_test]]]
ylist_control = [data1.iloc[:, [y_col_control]],
                 data2.iloc[:, [y_col_control]]]

x_col_circ = [1]
test_col = [0, 1]

test = ComparePCurve(Xlist, ylist_test, [
                     0, 1], circ_pos=x_col_circ, thresh=0.2, grid_size=[50, 50])
pickle.dump(test, open('vg_upgrade_test.dat', 'wb'))

control = ComparePCurve(Xlist, ylist_control, [
                        0, 1], circ_pos=x_col_circ, thresh=0.2, grid_size=[50, 50])
pickle.dump(control, open('vg_upgrade_control.dat', 'wb'))

VG_effect = test.weighted_diff - control.weighted_diff

print('VG effect: {:.2f}%'.format(VG_effect))

# print('Control statistics:')
# print('weightedDiff: {:.2f}%'.format(control.weighted_diff))
# print('weightedStatDiff: {:.2f}%'.format(control.weighted_stat_diff))
# print('scaledDiff: {:.2f}%'.format(control.scaled_diff))
# print('scaleddStatDiff: {:.2f}%'.format(control.scaled_stat_diff))
# print('unweightedDiff: {:.2f}%'.format(control.unweighted_diff))
# print('unweightedStatDiff: {:.2f}%'.format(control.unweighted_stat_diff))

# print('Test statistics:')
# print('weightedDiff: {:.2f}%'.format(test.weighted_diff))
# print('weightedStatDiff: {:.2f}%'.format(test.weighted_stat_diff))
# print('scaledDiff: {:.2f}%'.format(test.scaled_diff))
# print('scaledStatDiff: {:.2f}%'.format(test.scaled_stat_diff))
# print('unweightedDiff: {:.2f}%'.format(test.unweighted_diff))
# print('unweightedStatDiff: {:.2f}%'.format(test.unweighted_stat_diff))
