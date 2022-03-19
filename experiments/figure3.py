import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from dswe import covmatch


def nrd0(x):
    return 0.9 * min(np.std(x, ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349) * len(x)**(-0.2)


def approx_kde(X):
    var_range = np.linspace(min(X), max(X), 512).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=nrd0(X)
                        ).fit(X.reshape(-1, 1))
    var_density = np.exp(kde.score_samples(var_range))

    return var_range, var_density


data1 = pd.read_csv('../../../Downloads/DS/J77_Data&Code/Dataset2_1.csv')
data2 = pd.read_csv('../../../Downloads/DS/J77_Data&Code/Dataset2_2.csv')

xCol = [0, 3, 5]
thrs = [0.1, 0.1, 0.05]

data1 = data1.iloc[:, xCol].to_numpy()
data2 = data2.iloc[:, xCol].to_numpy()
Xlist = [data1, data2]

matched = covmatch.CovMatch(Xlist, thresh=thrs)

fig = plt.figure(figsize=(14, 20))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

ax = fig.add_subplot(3, 2, 1)
ax.set_title('Before Matching', fontsize=16)
X, y = approx_kde(data1[:, 0])
ax.plot(X, y, color='black')
X, y = approx_kde(data2[:, 0])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([0, 18])
ax.set_ylim([0, 0.23])
ax.set_xlabel('Wind Speed (m/s)', fontsize=14)
ax = fig.add_subplot(3, 2, 2)
ax.set_title('After Matching', fontsize=16)
X, y = approx_kde(matched.matched_data_X[0][:, 0])
ax.plot(X, y, color='black')
X, y = approx_kde(matched.matched_data_X[1][:, 0])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([0, 18])
ax.set_ylim([0, 0.23])
ax.set_xlabel('Wind Speed (m/s)', fontsize=14)

ax = fig.add_subplot(3, 2, 3)
ax.set_title('Before Matching', fontsize=16)
X, y = approx_kde(data1[:, 1])
ax.plot(X, y, color='black')
X, y = approx_kde(data2[:, 1])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([15, 35])
ax.set_ylim([0, 0.08])
ax.set_xlabel('Ambient Temperature ({}C)'.format(
    u'\N{DEGREE SIGN}'), fontsize=14)
ax = fig.add_subplot(3, 2, 4)
ax.set_title('After Matching', fontsize=16)
X, y = approx_kde(matched.matched_data_X[0][:, 1])
ax.plot(X, y, color='black')
X, y = approx_kde(matched.matched_data_X[1][:, 1])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([15, 35])
ax.set_ylim([0, 0.08])
ax.set_xlabel('Ambient Temperature ({}C)'.format(
    u'\N{DEGREE SIGN}'), fontsize=14)

ax = fig.add_subplot(3, 2, 5)
ax.set_title('Before Matching', fontsize=16)
X, y = approx_kde(data1[:, 2])
ax.plot(X, y, color='black')
X, y = approx_kde(data2[:, 2])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([0, 0.4])
ax.set_ylim([0, 8])
ax.set_xlabel('Turbulence Density', fontsize=14)
ax = fig.add_subplot(3, 2, 6)
ax.set_title('After Matching', fontsize=16)
X, y = approx_kde(matched.matched_data_X[0][:, 2])
ax.plot(X, y, color='black')
X, y = approx_kde(matched.matched_data_X[1][:, 2])
ax.plot(X, y, color='red', linestyle='--')
ax.set_xlim([0, 0.4])
ax.set_ylim([0, 8])
ax.set_xlabel('Turbulence Density', fontsize=14)

plt.show()

fig.savefig('figure_3.pdf')
