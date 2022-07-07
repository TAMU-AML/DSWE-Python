from numpy import float16
import pandas as pd
import pickle
from dswe.comparePCurve import ComparePCurve

data = pd.read_csv(
    '~/workspace/Datasets/Turbine Upgrade Dataset(Pitch Angle Pair, Table7.3).csv')
data1 = data.loc[data['upgrade.status'] == 0, :]
data2 = data.loc[data['upgrade.status'] == 1, :]

r = range(2, 10)
r_prime = [1.25, 1.87, 2.49, 3.11, 3.74, 4.36, 4.98, 5.60]

results = pd.DataFrame(index=['r_prime', 'Our estimate', 'Our estimate/r_prime'],
                       columns=[str(i) + '%' for i in r], dtype=float16)
results.iloc[0, :] = r_prime

x_col = [2, 3, 4, 5, 6]
y_col_control = 17
x_col_circ = [1]
test_col = [0, 1]

Xlist = [data1.iloc[:, x_col], data2.iloc[:, x_col]]
ylist_control = [data1.iloc[:, [y_col_control]],
                 data2.iloc[:, [y_col_control]]]

control = ComparePCurve(Xlist, ylist_control, [
                        0, 1], circ_pos=x_col_circ, thresh=0.2, grid_size=[50, 50])
pickle.dump(control, open('pitch_angle_control.dat', 'wb'))

for i, y_col_test in enumerate(range(9, 17)):
    ylist_test = [data1.iloc[:, [y_col_test]], data2.iloc[:, [y_col_test]]]
    test = ComparePCurve(Xlist, ylist_test, [
                         0, 1], circ_pos=x_col_circ, thresh=0.2, grid_size=[50, 50])
    pickle.dump(test, open('pitch_angle_test_' + str(i) + '.dat', 'wb'))
    results.iloc[1, i] = test.weighted_diff - control.weighted_diff

results.iloc[2, :] = results.iloc[1, :].to_numpy() / \
    results.iloc[0, :].to_numpy()
results.to_csv('pitch_angle_results.csv', float_format='{:.2f}'.format)
print(results.head())
