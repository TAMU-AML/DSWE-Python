import math
import numpy as np
from scipy.linalg import cholesky, solve_triangular, pinvh


def outer_diff(X1, X2):
    od = np.zeros((X1.shape[0], X2.shape[0]))
    x = X2.T
    for i in range(od.shape[1]):
        od[:, i] = X1
    for i in range(od.shape[0]):
        od[i, :] -= x

    return od


def compute_correl_mat(X1, X2, theta):
    corr_mat = np.zeros((X1.shape[0], X2.shape[0]))

    for i in range(len(theta)):
        corr_mat = corr_mat + \
            np.power(outer_diff(X1[:, i], X2[:, i]) / theta[i], 2)

    corr_mat = np.exp(-0.5 * corr_mat)

    return corr_mat


def compute_weighted_y(X, y, params):
    theta = params['theta']
    beta, sigma_f, sigma_n = params['beta'], params['sigma_f'], params['sigma_n']

    train_mat = math.pow(sigma_f, 2) * compute_correl_mat(X, X, theta)
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

    cov_mat = math.pow(sigma_f, 2) * compute_correl_mat(X, X, theta)
    diag_idx = np.diag_indices(cov_mat.shape[0])
    cov_mat[diag_idx] += math.pow(sigma_n, 2)

    upper_chol_mat = cholesky(cov_mat)
    y_dash = y - beta

    t1 = 0.5 * np.dot(y_dash.T, solve_triangular(upper_chol_mat,
                                                 solve_triangular(upper_chol_mat.T, y_dash, lower=True)))
    t2 = np.log(np.abs(upper_chol_mat.diagonal())).sum()
    t3 = np.log(2 * math.pi) * upper_chol_mat.shape[0] / 2

    return t1 + t2 + t3


def predict_GP(train_X, weighted_y, test_X, params):
    pred = np.zeros((test_X.shape[0], 1))
    theta, sigma_f, beta = params['theta'], params['sigma_f'], params['beta']
    test_cov_mat = math.pow(sigma_f, 2) * \
        compute_correl_mat(train_X, test_X, theta)
    pred = beta + np.dot(test_cov_mat, weighted_y)

    return pred


def compute_loglike_grad_GP(X, y, params):
    theta = params['theta']
    n_theta = len(theta)
    beta, sigma_f, sigma_n = params['beta'], params['sigma_f'], params['sigma_n']

    correl_mat = compute_correl_mat(X, X, theta)
    cov_mat = math.pow(sigma_f, 2) * correl_mat
    diag_idx = np.diag_indices(cov_mat.shape[0])
    cov_mat[diag_idx] += math.pow(sigma_n, 2)

    inv_mat = pinvh(cov_mat)
    grad_val = np.zeros(n_theta + 3)

    y_dash = y - beta
    alpha = np.dot(inv_mat, y_dash)
    diff_mat = np.dot(alpha.reshape(-1, 1), alpha.reshape(-1, 1).T) - inv_mat
    onevec = np.ones(len(y))
    sol_onevec = np.dot(inv_mat, onevec)

    for i in range(n_theta):
        del_theta_mat = (math.pow(
            sigma_f, 2) * (np.power(outer_diff(X[:, i], X[:, i]), 2) / math.pow(theta[i], 3))) * correl_mat
        del_theta_mat = np.dot(diff_mat, del_theta_mat)
        grad_val[i] = -0.5 * np.trace(del_theta_mat)

    del_sigma_f_mat = 2 * sigma_f * correl_mat
    del_sigma_f_mat = np.dot(diff_mat, del_sigma_f_mat)
    grad_val[n_theta] = -0.5 * np.trace(del_sigma_f_mat)
    del_sigma_n_mat = 2 * sigma_n * diff_mat
    grad_val[n_theta + 1] = -0.5 * np.trace(del_sigma_n_mat)
    grad_val[n_theta + 2] = 0.5 * (2 * beta * np.dot(onevec.T, sol_onevec) - np.dot(
        y.T, sol_onevec) - np.dot(onevec.T, alpha + (beta * sol_onevec)))

    return grad_val


def compute_loglike_grad_GP_zero_mean(X, y, params):
    theta = params['theta']
    n_theta = len(theta)
    beta, sigma_f, sigma_n = params['beta'], params['sigma_f'], params['sigma_n']

    correl_mat = compute_correl_mat(X, X, theta)
    cov_mat = math.pow(sigma_f, 2) * correl_mat
    diag_idx = np.diag_indices(cov_mat.shape[0])
    cov_mat[diag_idx] += math.pow(sigma_n, 2)

    inv_mat = pinvh(cov_mat)
    grad_val = np.zeros(n_theta + 2)

    y_dash = y - beta
    alpha = np.dot(inv_mat, y_dash)
    diff_mat = np.dot(alpha.reshape(-1, 1), alpha.reshape(-1, 1).T) - inv_mat
    onevec = np.ones(len(y))
    sol_onevec = np.dot(inv_mat, onevec)

    for i in range(n_theta):
        del_theta_mat = (math.pow(
            sigma_f, 2) * (np.power(outer_diff(X[:, i], X[:, i]), 2) / math.pow(theta[i], 3))) * correl_mat
        del_theta_mat = np.dot(diff_mat, del_theta_mat)
        grad_val[i] = -0.5 * np.trace(del_theta_mat)

    del_sigma_f_mat = 2 * sigma_f * correl_mat
    del_sigma_f_mat = np.dot(diff_mat, del_sigma_f_mat)
    grad_val[n_theta] = -0.5 * np.trace(del_sigma_f_mat)
    del_sigma_n_mat = 2 * sigma_n * diff_mat
    grad_val[n_theta + 1] = -0.5 * np.trace(del_sigma_n_mat)

    return grad_val
