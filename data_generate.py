import numpy as np
import copy

def tmc_data_generate(data_len, start_prob, u_trans_matrix, yu_matrix,
                      xv_mean, xv_cov, seq_num=1, seeds=1024):
    start_prob_cdf = np.cumsum(start_prob)
    trans_matrix_cdf = np.cumsum(u_trans_matrix, axis=1)
    yu_matrix_cdf = np.cumsum(yu_matrix)

    u_list = np.empty([seq_num, data_len])
    y_list = np.empty([seq_num, data_len])
    x_list = np.empty([seq_num, data_len])
    for i in range(seq_num):
        random_state = np.random.RandomState(seeds + 10 * i)
        current_state = (start_prob_cdf > random_state.rand()).argmax()
        u = [current_state]
        y = [(yu_matrix_cdf[current_state] > random_state.rand()).argmax()]
        x = [_generate_sample_from_state(
            (u[0], y[0]), xv_mean, xv_cov, random_state)]

        for t in range(data_len - 1):
            current_state = (trans_matrix_cdf[current_state] >
                             random_state.rand()).argmax()
            u.append(current_state)
            y.append((yu_matrix_cdf[current_state] > random_state.rand()).argmax())
            x.append(_generate_sample_from_state(
                (u[t + 1], y[t + 1]), xv_mean, xv_cov, random_state))

        u_list[i] = copy.deepcopy(np.array(u, dtype='int'))
        y_list[i] = copy.deepcopy(np.array(y, dtype='int'))
        x_list[i] = copy.deepcopy(np.array(x))

    return u_list, y_list, x_list

def _generate_sample_from_state(state, means, covs, random_state):
    return random_state.normal(means[state], np.sqrt(covs[state]))

def initial_generate(n, start_prob, trans_matrix, seeds=1024):
    random_state = np.random.RandomState(seeds)

    start_prob_cdf = np.cumsum(start_prob)
    trans_matrix_cdf = np.cumsum(trans_matrix, axis=1)

    current_state = (start_prob_cdf > random_state.rand()).argmax()
    y = [current_state]

    for t in range(n - 1):
        current_state = (trans_matrix_cdf[current_state] >
                         random_state.rand()).argmax()
        y.append(current_state)

    return np.array(y)
