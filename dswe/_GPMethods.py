import math
import numpy as np
from scipy.linalg import cholesky, solve_triangular, pinvh, eigh
from scipy import stats


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

    t1 = 0.5 * np.matmul(y_dash.T, solve_triangular(upper_chol_mat,
                                                    solve_triangular(upper_chol_mat.T, y_dash, lower=True)))
    t2 = np.log(np.abs(upper_chol_mat.diagonal())).sum()
    t3 = np.log(2 * math.pi) * upper_chol_mat.shape[0] / 2

    return t1 + t2 + t3


def predict_GP(train_X, weighted_y, test_X, params):
    pred = np.zeros((test_X.shape[0], 1))
    theta, sigma_f, beta = params['theta'], params['sigma_f'], params['beta']
    test_cov_mat = math.pow(sigma_f, 2) * \
        compute_correl_mat(test_X, train_X, theta)
    pred = beta + np.matmul(test_cov_mat, weighted_y)

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
    alpha = np.matmul(inv_mat, y_dash)
    diff_mat = np.matmul(alpha.reshape(-1, 1),
                         alpha.reshape(-1, 1).T) - inv_mat
    onevec = np.ones(len(y))
    sol_onevec = np.matmul(inv_mat, onevec)

    for i in range(n_theta):
        del_theta_mat = (math.pow(
            sigma_f, 2) * (np.power(outer_diff(X[:, i], X[:, i]), 2) / math.pow(theta[i], 3))) * correl_mat
        del_theta_mat = np.matmul(diff_mat, del_theta_mat)
        grad_val[i] = -0.5 * np.trace(del_theta_mat)

    del_sigma_f_mat = 2 * sigma_f * correl_mat
    del_sigma_f_mat = np.matmul(diff_mat, del_sigma_f_mat)
    grad_val[n_theta] = -0.5 * np.trace(del_sigma_f_mat)
    del_sigma_n_mat = 2 * sigma_n * diff_mat
    grad_val[n_theta + 1] = -0.5 * np.trace(del_sigma_n_mat)
    grad_val[n_theta + 2] = 0.5 * (2 * beta * np.matmul(onevec.T, sol_onevec) - np.matmul(
        y.T, sol_onevec) - np.matmul(onevec.T, alpha + (beta * sol_onevec)))

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
    alpha = np.matmul(inv_mat, y_dash)
    diff_mat = np.matmul(alpha.reshape(-1, 1),
                         alpha.reshape(-1, 1).T) - inv_mat
    onevec = np.ones(len(y))
    sol_onevec = np.matmul(inv_mat, onevec)

    for i in range(n_theta):
        del_theta_mat = (math.pow(
            sigma_f, 2) * (np.power(outer_diff(X[:, i], X[:, i]), 2) / math.pow(theta[i], 3))) * correl_mat
        del_theta_mat = np.matmul(diff_mat, del_theta_mat)
        grad_val[i] = -0.5 * np.trace(del_theta_mat)

    del_sigma_f_mat = 2 * sigma_f * correl_mat
    del_sigma_f_mat = np.matmul(diff_mat, del_sigma_f_mat)
    grad_val[n_theta] = -0.5 * np.trace(del_sigma_f_mat)
    del_sigma_n_mat = 2 * sigma_n * diff_mat
    grad_val[n_theta + 1] = -0.5 * np.trace(del_sigma_n_mat)

    return grad_val


def compute_diff_conv_(X1, y1, X2, y2, XT, theta, sigma_f, sigma_n, beta):
    KX1X1 = math.pow(sigma_f, 2) * compute_correl_mat(X1, X1, theta)
    diag_idx = np.diag_indices(KX1X1.shape[0])
    KX1X1[diag_idx] += math.pow(sigma_n, 2)
    inv_KX1X1 = pinvh(KX1X1)
    KX1X1 = None

    KXTX1 = math.pow(sigma_f, 2) * compute_correl_mat(XT, X1, theta)
    mu1 = beta + np.matmul(np.matmul(KXTX1, inv_KX1X1), (y1 - beta))
    K1 = np.matmul(inv_KX1X1, KXTX1.T)
    K = np.matmul(KXTX1, K1)
    KXTX1 = None
    inv_KX1X1 = None

    KX2X2 = math.pow(sigma_f, 2) * compute_correl_mat(X2, X2, theta)
    diag_idx = np.diag_indices(KX2X2.shape[0])
    KX2X2[diag_idx] += math.pow(sigma_n, 2)
    inv_KX2X2 = pinvh(KX2X2)
    KX2X2 = None

    KXTX2 = math.pow(sigma_f, 2) * compute_correl_mat(XT, X2, theta)
    mu2 = beta + np.matmul(np.matmul(KXTX2, inv_KX2X2), (y2 - beta))
    K2 = np.matmul(KXTX2, inv_KX2X2)
    inv_KX2X2 = None
    K = K + np.matmul(K2, KXTX2.T)
    KXTX2 = None

    KX2X1 = math.pow(sigma_f, 2) * compute_correl_mat(X2, X1, theta)
    K = K - (2 * np.matmul(np.matmul(K2, KX2X1), K1))
    K = (K + K.T) / 2

    return {'diff_cov_mat': K, 'mu1': mu1, 'mu2': mu2}


def compute_conf_band_(diff_cov_mat, conf_level):
    eig_val, eig_vec = eigh(diff_cov_mat)
    first_idx = np.where(eig_val[eig_val < 1e-8])[0][-1]
    lambdaa = np.diag(np.flip(eig_val[first_idx:len(eig_val)]))
    eig_mat = np.fliplr(eig_vec[:, first_idx:len(eig_val)])
    n_eig = lambdaa.shape[0]
    radius = np.sqrt(stats.chi2.ppf(conf_level, n_eig))
    n_samples = 1000
    Z = np.zeros((n_eig, n_samples))

    n = 0
    while n < n_samples:
        z_sample = np.random.randn(n_eig)
        z_sum = np.sqrt(np.power(z_sample, 2).sum())

        if z_sum <= radius:
            Z[:, n] = z_sample
            n += 1

    G = np.matmul(np.matmul(eig_mat, np.sqrt(lambdaa)), Z)
    G = np.abs(G)
    band = G.max(axis=1)

    return band
