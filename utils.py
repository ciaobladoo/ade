import numpy as np
from scipy.spatial.distance import cdist


def get_gamma(X, bandwidth):
    x_norm = np.sum(X ** 2, axis=1, keepdims=True)
    x_t = X.transpose()
    x_norm_t = np.reshape(x_norm, (1, -1))
    t = x_norm + x_norm_t - 2.0 * np.matmul(X, x_t)
    d = np.maximum(t, 0)

    d = d[np.isfinite(d)]
    d = d[d > 0]
    median_dist2 = float(np.median(d))
    print('median_dist2:', median_dist2)
    gamma = 0.5 / median_dist2 / bandwidth
    return gamma


def get_kernel_mat(x, landmarks, gamma):
    feat_dim = x.shape[1]
    batch_size = x.shape[0]
    d = cdist(x, landmarks, metric='sqeuclidean')

    # get kernel matrix
    k = np.exp(d * -gamma)
    k = np.reshape(k, (batch_size, -1))
    return k


def MMD(y, x, gamma):
    kxx = get_kernel_mat(x, x, gamma)
    np.fill_diagonal(kxx, 0)
    kxx = np.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    np.fill_diagonal(kyy, 0)
    kyy = np.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = np.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd
