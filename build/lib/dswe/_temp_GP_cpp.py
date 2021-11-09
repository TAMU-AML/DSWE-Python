import math
import numpy as np
from scipy.linalg import cholesky, solve_triangular


def outer_diff(X1, X2):
    od = np.zeros((X1.shape[0], X2.shape[0]))
    x = X2.T
    for i in range(od.shape[1]):
        od[:, i] = X1
    for i in range(od.shape[0]):
        od[i, :] -= x

    return od


def compute_correl_mat(X1, X2, theta):
    corr_mat = np.zeroes((X1.shape[0], X2.shape[0]))

    for i in range(len(theta)):
        corr_mat = corr_mat + \
            np.pow(outer_diff(X1[:, i], X2[:, i])/theta[i], 2)

    corr_mat = np.exp(-0.5*corr_mat)

    return corr_mat


def compute_weighted_y(X, y, params):
    theta = params['theta']
    beta, sigma_f, sigma_n = params['beta'], params['sigma_f'], params['sigma_n']

    train_mat = math.pow(sigma_f, 2)*compute_correl_mat(X, X, theta)
    diag_idx = np.diag_indices(train_mat.shape[0])
    train_mat[diag_idx] += math.pow(sigma_n, 2)

    upper_chol_train_mat = cholesky(train_mat)
    y_dash = y - beta
    weighted_y = solve_triangular(upper_chol_train_mat, solve_triangular(
        upper_chol_train_mat.T, y_dash, lower=True))

    return weighted_y


def compute_loglike_GP(X, y, params):
    theta = params['theta']
    beta, sigma_f, sigma_n = params['beta'], params['sigma_f'], params['sigma_n']

    cov_mat = math.pow(sigma_f, 2)*compute_correl_mat(X, X, theta)
    diag_idx = np.diag_indices(cov_mat.shape[0])
    cov_mat[diag_idx] += math.pow(sigma_n, 2)

    upper_chol_mat = cholesky(cov_mat)
    y_dash = y - beta

    t1 = 0.5*np.asscalar(y_dash.T*solve_triangular(upper_chol_mat,
                         solve_triangular(upper_chol_mat.T, y_dash, lower=True)))
    t2 = np.trace(np.log(np.abs(upper_chol_mat.diagonal())))
    t3 = np.log(2*math.pi)*upper_chol_mat.shape[0]/2

    return t1+t2+t3
