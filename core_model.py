import numpy as np
import copy
from scipy.special import psi
from basic_algorithm import nix_posterior, nix_posterior_vb, \
    forward_backward, forward_backward_no_para, \
    viterbi, viterbi_no_para

def BayesEM(y0, x, start_prob, dir_prior, nix_mean, nix_hyper, max_step, threshold):
    state_num = len(start_prob)
    p = np.empty([state_num, state_num])
    mu = np.empty(state_num)
    sigma2 = np.empty(state_num)
    dir_row_sum = np.sum(dir_prior, axis=1)
    (k0, v0, t02) = nix_hyper

    # Initial step
    xi = np.zeros([state_num, state_num])
    for t in range(len(y0) - 1):
        l, j = y0[t], y0[t+1]
        xi[l, j] += 1

    xi_row_sum = np.sum(xi, axis=1)
    for l in range(state_num):
        for j in range(state_num):
            p[l, j] = (xi[l, j] + dir_prior[l, j] - 1) / \
                      (xi_row_sum[l] + dir_row_sum[l] - state_num)
        x_sk = x[np.where(y0 == l)[0]]
        muk, _, vk, tk2 = nix_posterior(x_sk, nix_mean[l], k0, v0, t02)
        mu[l] = muk
        sigma2[l] = (vk * tk2) / (vk + 2)

    gamma, xi = forward_backward(x, start_prob, p, mu, sigma2)
    last_p, last_mu, last_sigma2 = \
        copy.deepcopy(p), copy.deepcopy(mu), copy.deepcopy(sigma2)

    # Update step
    for step in range(max_step):
        # Update transition matrix
        xi_row_sum = np.sum(xi, axis=1)
        for l in range(state_num):
            for j in range(state_num):
                p[l, j] = (xi[l, j] + dir_prior[l, j] - 1) / \
                          (xi_row_sum[l] + dir_row_sum[l] - state_num)

        # Update emission parameters
        for k in range(state_num):
            up_sum, down_sum = 0, 0
            for t in range(len(x)):
                up_sum += x[t] * gamma[t, k]
                down_sum += gamma[t, k]
            mu[k] = (up_sum + nix_mean[k] * k0) / (down_sum + k0)

            up_sum = 0
            for t in range(len(x)):
                up_sum += np.square(x[t] - mu[k]) * gamma[t, k]
            sigma2[k] = (v0 * t02 + up_sum + np.square(mu[k] - nix_mean[k]) * k0) / \
                        (down_sum + v0 + 3)

        # Compute parameters gap
        current_gap = np.linalg.norm(p - last_p)
        current_gap += np.linalg.norm(mu - last_mu)
        current_gap += np.linalg.norm(sigma2 - last_sigma2)
        if current_gap <= threshold:
            break

        gamma, xi = forward_backward(x, start_prob, p, mu, sigma2)
        last_p, last_mu, last_sigma2 = \
            copy.deepcopy(p), copy.deepcopy(mu), copy.deepcopy(sigma2)

    return viterbi(x, start_prob, p, mu, sigma2)

def VBEM(y0, x, start_prob, dir_prior, nix_mean, nix_hyper, max_step, threshold):
    state_num = len(start_prob)
    u = np.empty([state_num, state_num])
    h = np.empty([state_num, len(x)])
    dir_row_sum = np.sum(dir_prior, axis=1)
    (k0, v0, t02) = nix_hyper

    # Initial step
    xi = np.zeros([state_num, state_num])
    for t in range(len(y0) - 1):
        l, j = y0[t], y0[t + 1]
        xi[l, j] += 1

    xi_row_sum = np.sum(xi, axis=1)
    for l in range(state_num):
        tmp_value = psi(dir_row_sum[l] + xi_row_sum[l])
        for j in range(state_num):
            u[l, j] = np.exp(psi(dir_prior[l, j] + xi[l, j]) - tmp_value)

        x_sk = x[np.where(y0 == l)[0]]
        muk, kk, vk, tk2 = nix_posterior(x_sk, nix_mean[l], k0, v0, t02)
        for t in range(len(x)):
            h[l, t] = np.exp(-np.log(2*np.pi*tk2) / 2 - (np.log(vk/2) - psi(vk/2)) / 2 - \
                      np.square(x[t]) / (2*tk2) + x[t] * muk / tk2 - \
                      (1/kk + np.square(muk/tk2)) / 2)

    gamma, xi = forward_backward_no_para(x, start_prob, u, h)
    last_u, last_h = copy.deepcopy(u), copy.deepcopy(h)

    for step in range(max_step):
        # Update transition matrix
        xi_row_sum = np.sum(xi, axis=1)
        for l in range(state_num):
            tmp_value = psi(dir_row_sum[l] + xi_row_sum[l])
            for j in range(state_num):
                u[l, j] = np.exp(psi(dir_prior[l, j] + xi[l, j]) - tmp_value)

        # Update emission parameters
        for k in range(state_num):
            muk, kk, vk, tk2 = \
                nix_posterior_vb(x, nix_mean[k], k0, v0, t02, gamma[:, k])
            for t in range(len(x)):
                h[k, t] = np.exp(-np.log(2 * np.pi * tk2) / 2 - (np.log(vk / 2) - psi(vk / 2)) / 2 - \
                          np.square(x[t]) / (2 * tk2) + x[t] * muk / tk2 - \
                          (1 / kk + np.square(muk / tk2)) / 2)

        # Compute parameters gap
        current_gap = np.linalg.norm(u - last_u)
        current_gap += np.linalg.norm(h - last_h)
        if current_gap <= threshold:
            break

        gamma, xi = forward_backward_no_para(x, start_prob, u, h)
        last_u, last_h = copy.deepcopy(u), copy.deepcopy(h)

    return viterbi_no_para(x, start_prob, u, h)

def SegmentationMM(y0, x, start_prob, dir_prior, nix_mean, nix_hyper, max_step, threshold):
    state_num = len(start_prob)
    p = np.empty([state_num, state_num])
    mu = np.empty(state_num)
    sigma2 = np.empty(state_num)
    dir_row_sum = np.sum(dir_prior, axis=1)
    (k0, v0, t02) = nix_hyper

    y = copy.deepcopy(y0)
    last_p, last_mu, last_sigma2 = \
        copy.deepcopy(p), copy.deepcopy(mu), copy.deepcopy(sigma2)

    for step in range(max_step):
        # Update transition parameter
        n = np.zeros([state_num, state_num])
        for t in range(len(y) - 1):
            l, j = y[t], y[t + 1]
            n[l, j] += 1

        n_row_sum = np.sum(n, axis=1)
        for l in range(state_num):
            tmp_value = dir_row_sum[l] + n_row_sum[l] - state_num
            for j in range(state_num):
                p[l, j] = (dir_prior[l ,j] + n[l, j] - 1) / tmp_value

        # Update emission parameters
        for k in range(state_num):
            x_sk = x[np.where(y == k)[0]]
            muk, _, vk, tk2 = nix_posterior(x_sk, nix_mean[k], k0, v0, t02)

            mu[k] = muk
            sigma2[k] = (vk * tk2) / (vk + 2)

        # Compute parameters gap
        y = viterbi(x, start_prob, p, mu, sigma2)

        current_gap = np.linalg.norm(p - last_p)
        current_gap += np.linalg.norm(mu - last_mu)
        current_gap += np.linalg.norm(sigma2 - last_sigma2)
        if current_gap <= threshold:
            break

        last_p, last_mu, last_sigma2 = \
            copy.deepcopy(p), copy.deepcopy(mu), copy.deepcopy(sigma2)

    return y

def SegmentationEM(y0, x, start_prob, dir_prior, nix_mean, nix_hyper, max_step, threshold):
    state_num = len(start_prob)
    u = np.empty([state_num, state_num])
    h = np.empty([state_num, len(x)])
    dir_row_sum = np.sum(dir_prior, axis=1)
    (k0, v0, t02) = nix_hyper

    y = copy.deepcopy(y0)
    last_u, last_h = copy.deepcopy(u), copy.deepcopy(h)

    for step in range(max_step):
        # Update transition parameters
        n = np.zeros([state_num, state_num])
        for t in range(len(y) - 1):
            l, j = y[t], y[t + 1]
            n[l, j] += 1

        n_row_sum = np.sum(n, axis=1)
        for l in range(state_num):
            tmp_value = psi(dir_row_sum[l] + n_row_sum[l])
            for j in range(state_num):
                u[l, j] = np.exp(psi(dir_prior[l, j] + n[l, j]) - tmp_value)

        # Update emission parameters
        for k in range(state_num):
            x_sk = x[np.where(y == k)[0]]
            muk, kk, vk, tk2 = \
                nix_posterior(x_sk, nix_mean[k], k0, v0, t02)
            for t in range(len(x)):
                h[k, t] = np.exp(-np.log(2 * np.pi * tk2) / 2 - (np.log(vk / 2) - psi(vk / 2)) / 2 - \
                          np.square(x[t]) / (2 * tk2) + x[t] * muk / tk2 - \
                          (1 / kk + np.square(muk / tk2)) / 2)

        # Compute parameters gap
        y = viterbi_no_para(x, start_prob, u, h)

        current_gap = np.linalg.norm(u - last_u)
        current_gap += np.linalg.norm(h - last_h)
        if current_gap <= threshold:
            break

        last_u, last_h = copy.deepcopy(u), copy.deepcopy(h)

    return y
