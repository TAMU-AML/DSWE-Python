import numpy as np
import pandas as pd
from statsmodels.tsa import stattools as ts
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from dswe.covmatch import CovMatch
from dswe.funGP import FunGP


data1 = pd.read_csv('../../Downloads/DS/J77_Data&Code/Dataset3_1.csv')
data2 = pd.read_csv('../../Downloads/DS/J77_Data&Code/Dataset3_2.csv')

xCol = [1, 3, 7]

Xlist = [data2.iloc[:, xCol].to_numpy(), data1.iloc[:, xCol].to_numpy()]
ylist = [data2.iloc[:, 5].to_numpy(), data1.iloc[:, 5].to_numpy()]

matched_dataset1 = CovMatch(Xlist, ylist)
testset1 = np.linspace(
    1.025 * matched_dataset1.min_max_matched['min'][0], 0.975 * matched_dataset1.min_max_matched['max'][0], 1000)
matched_data1 = matched_dataset1.matched_data_X

X1 = [matched_data1[0][:, 0].reshape(-1, 1),
      matched_data1[1][:, 0].reshape(-1, 1)]
y1 = [matched_dataset1.matched_data_y[0], matched_dataset1.matched_data_y[1]]

fgp = FunGP(X1, y1, testset1)
mu_diff1 = -fgp.mu_diff
band1 = fgp.band

shade_min = np.where(mu_diff1 - band1 >= 0.001)[0].min()
shade_max = np.where(mu_diff1 - band1 >= 0.001)[0].max()
shade_idx = list(range(shade_min, shade_max + 1))
np.random.seed(1)
sample_idx = np.random.choice(len(data2), len(data2), replace=False)

idx1 = int(len(data2) / 2)
data2_1 = data2.iloc[sample_idx[:idx1], :]
data2_2 = data2.iloc[sample_idx[idx1:], :]

Xlist2 = [data2_1.iloc[:, xCol].to_numpy(), data2_2.iloc[:, xCol].to_numpy()]
ylist2 = [data2_1.iloc[:, 5].to_numpy(), data2_2.iloc[:, 5].to_numpy()]

matched_dataset2 = CovMatch(Xlist2, ylist2)
testset2 = np.linspace(
    1.025 * matched_dataset2.min_max_matched['min'][0], 0.975 * matched_dataset2.min_max_matched['max'][0], 1000)
matched_data2 = matched_dataset2.matched_data_X

X2 = [matched_data2[0][:, 0].reshape(-1, 1),
      matched_data2[1][:, 0].reshape(-1, 1)]
y2 = [matched_dataset2.matched_data_y[0], matched_dataset2.matched_data_y[1]]

fgp2 = FunGP(X2, y2, testset2)
mu_diff2 = -fgp2.mu_diff
band2 = fgp2.band


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

ax1.set_title('f1 != f2', fontsize=16)
ax1.plot(testset1, mu_diff1, color='blue', label='Difference')
ax1.plot(testset1, band1, color='red',
         linestyle='--', label='95% Confidence band')
ax1.plot(testset1, -band1, color='red', linestyle='--')
ax1.fill_between(testset1.squeeze(), mu_diff1, band1, where=mu_diff1 >= band1,
                 facecolor='deepskyblue', interpolate=True, label='Statistical Difference')
ax1.axhline(y=0, color='black', linestyle='--')
ax1.set_xlim([0, 20])
ax1.set_ylim([-0.25, 0.25])
ax1.set_xlabel('Wind speed (m/s)', fontsize=14)
ax1.set_ylabel('Difference of normalized power', fontsize=14)
ax1.legend(loc='upper left', fontsize=12)

ax2.set_title('f1 = f2', fontsize=16)
ax2.plot(testset2, mu_diff2, color='blue', label='Difference')
ax2.plot(testset2, band2, color='red',
         linestyle='--', label='95% Confidence band')
ax2.plot(testset2, -band2, color='red', linestyle='--')
ax2.axhline(y=0, color='black', linestyle='--')
ax2.set_xlim([0, 20])
ax2.set_ylim([-0.25, 0.25])
ax2.set_xlabel('Wind speed (m/s)', fontsize=14)
ax2.set_ylabel('Difference of normalized power', fontsize=14)
ax2.legend(loc='upper left', fontsize=12)

plt.show()

fig.savefig('figure_4.pdf')
