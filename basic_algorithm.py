import numpy as np
import scipy.stats as st
from scipy.special import logsumexp

def nix_posterior(x, xi, k0, v0, t02):
    n = len(x)
    if n == 0:
        return xi, k0, v0, t02
    x_mean = np.mean(x)
    x_var = 0
    for t in x:
        x_var += np.square(t - x_mean)

    kn = k0 + n
    vn = v0 + n
    mun = (k0 * xi + n * x_mean) / kn
    tn2 = v0 * t02 + x_var + \
          (k0 * n * np.square(x_mean - xi)) / kn
    tn2 /= vn

    return mun, kn, vn, tn2

def nix_posterior_vb(x, xi, k0, v0, t02, gamma):
    n = len(x)
    g = np.sum(gamma)
    x_slide = 0
    for t in range(n):
        x_slide += gamma[t] * x[t]
    x_slide /= g

    kn = k0 + g
    vn = v0 + g
    mun = (k0 * xi + g * x_slide) / kn
    tn2 = 0
    for t in range(n):
        tn2 += np.square(x[t] - x_slide) * gamma[t]
    tn2 += v0 * t02 + (k0 * g * np.square(x_slide - xi)) / (k0 + g)
    tn2 /= vn

    return mun, kn, vn, tn2

def forward_backward(x, start_prob, trans_matrix, means, vars):
    state_num = len(means)

    # Compute values
    log_h = np.zeros([state_num, len(x)])

    for t in range(len(x)):
        for k in range(state_num):
            log_h[k, t] = \
                np.log(st.norm.pdf(x[t], loc=means[k], scale=np.sqrt(vars[k])))

    log_start_prob = _log_mask_zero(start_prob)
    log_trans_matrix = _log_mask_zero(trans_matrix)

    return _forward_backward(x, log_start_prob, log_trans_matrix, log_h)

def forward_backward_no_para(x, start_prob, trans_matrix, h):
    log_start_prob = _log_mask_zero(start_prob)
    log_trans_matrix = _log_mask_zero(trans_matrix)
    log_h = _log_mask_zero(h)

    return _forward_backward(x, log_start_prob, log_trans_matrix, log_h)

def viterbi(x, start_prob, trans_matrix, means, vars):
    state_num = len(means)

    # Compute values
    log_h = np.zeros([state_num, len(x)])

    for t in range(len(x)):
        for k in range(state_num):
            log_h[k, t] = \
                np.log(st.norm.pdf(x[t], loc=means[k], scale=np.sqrt(vars[k])))

    log_start_prob = _log_mask_zero(start_prob)
    log_trans_matrix = _log_mask_zero(trans_matrix)

    return _viterbi(x, log_start_prob, log_trans_matrix, log_h)

def viterbi_no_para(x, start_prob, trans_matrix, h):
    log_start_prob = _log_mask_zero(start_prob)
    log_trans_matrix = _log_mask_zero(trans_matrix)
    log_h = _log_mask_zero(h)

    return _viterbi(x, log_start_prob, log_trans_matrix, log_h)

def _forward_backward(x, log_start_prob, log_trans_matrix, log_h):
    state_num = log_trans_matrix.shape[0]
    log_alpha = np.empty([len(x), state_num])
    log_beta = np.empty([len(x), state_num])

    # Forward
    work_buffer = np.zeros(state_num)

    for k in range(state_num):
        log_alpha[0][k] = log_start_prob[k] + log_h[k, 0]

    for t in range(len(x) - 1):
        for k in range(state_num):
            for j in range(state_num):
                work_buffer[j] = log_alpha[t][j] + log_trans_matrix[j, k]
            log_alpha[t + 1][k] = _log_sum_exp(work_buffer) + log_h[k, t + 1]

    # Backward
    work_buffer = np.zeros(state_num)

    log_beta[-1] = np.zeros(state_num)

    for t in range(len(x) - 2, -1, -1):
        for k in range(state_num):
            for j in range(state_num):
                work_buffer[j] = log_trans_matrix[k, j] + \
                                 log_beta[t + 1][j] + \
                                 log_h[j, t + 1]
            log_beta[t][k] = _log_sum_exp(work_buffer)

    # Gamma
    log_gamma = log_alpha + log_beta
    _log_normalize(log_gamma, 1)

    # Xi
    log_xi_sum = np.full([state_num, state_num], -np.inf)
    work_buffer = np.full([state_num, state_num], -np.inf)

    log_x_prob = _log_sum_exp(log_alpha[-1])

    for t in range(len(x) - 1):
        for l in range(state_num):
            for j in range(state_num):
                work_buffer[l, j] = log_alpha[t, l] + \
                                    log_trans_matrix[l, j] + \
                                    log_beta[t + 1, j] + \
                                    log_h[j, t + 1] - \
                                    log_x_prob

        for l in range(state_num):
            for j in range(state_num):
                log_xi_sum[l, j] = _log_add_exp(log_xi_sum[l, j],
                                                work_buffer[l, j])

    with np.errstate(under="ignore"):
        return np.exp(log_gamma), np.exp(log_xi_sum)

def _viterbi(x, log_start_prob, log_trans_matrix, log_h):
    state_num = log_trans_matrix.shape[0]
    log_delta = np.zeros([len(x), state_num])
    y = np.empty(len(x), dtype='int')
    work_buffer = np.empty(state_num)

    for k in range(state_num):
        log_delta[0, k] = log_start_prob[k] + log_h[k, 0]

    for t in range(len(x) - 1):
        for k in range(state_num):
            for l in range(state_num):
                work_buffer[l] = log_delta[t, l] + \
                                 log_trans_matrix[l, k]
            log_delta[t+1, k] = np.max(work_buffer) + log_h[k, t+1]

    y[-1] = np.argmax(log_delta[-1])

    for t in range(len(x) - 2, -1, -1):
        for k in range(state_num):
            work_buffer[k] = log_delta[t, k] + \
                             log_trans_matrix[k, y[t+1]]

        y[t] = np.argmax(work_buffer)

    return y

def _log_mask_zero(a):
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)

def _log_sum_exp(x):
    x_max = np.max(x)
    if np.isinf(x_max):
        return -np.Infinity

    acc = 0
    for k in range(len(x)):
        acc += np.exp(x[k] - x_max)

    return np.log(acc) + x_max

def _log_normalize(a, axis=None):
    with np.errstate(under="ignore"):
        a_lse = logsumexp(a, axis, keepdims=True)
    a -= a_lse

def _log_add_exp(a, b):
    if np.isinf(a) and a < 0:
        return b
    elif np.isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + np.log1p(np.exp(-np.fabs(a - b)))

