import numpy as np
from core_model import BayesEM, VBEM, SegmentationMM, SegmentationEM
from data_generate import tmc_data_generate, initial_generate

n = 600

u_trans_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])

yu1 = np.array([[0.7, 0.3], [0.3, 0.7]])
yu2 = np.array([[0.7, 0.3], [0.3, 0.7]])
yu3 = np.array([[0.8, 0.2], [0.2, 0.8]])
yu_matrix = np.array([yu1, yu2, yu3])

xv_m1 = np.array([[0, 2.5], [2.5, 5]])
xv_m2 = np.array([[0, 0.4], [1.6, 2]])
xv_m3 = np.array([[0, 0.8], [1.2, 2]])
xv_mean = np.array([xv_m1, xv_m2, xv_m3])

xv_cov = np.array([[1, 1], [1, 1]])

initial_para = [0.25, 0.5, 0.75]
start_prob = np.array([0.5, 0.5])

Q1 = np.array([[0.5, 0.5], [0.5, 0.5]])
Q2 = np.array([[0.6, 0.4], [0.4, 0.6]])
Q3 = np.array([[0.4, 0.6], [0.6, 0.4]])
Q = np.array([Q1, Q2, Q3])
M = [600, 150, 50, 10, 5]

nix_mean = np.array([0.8, 2.0])
nix_hyper = np.array([10, 50, 0.25])

for case in range(3):
    print('Now is case ', case)
    seq_num = 20
    u_list, y_list, x_list = tmc_data_generate(n, start_prob, u_trans_matrix,
                                               yu_matrix[case], xv_mean[case],
                                               xv_cov, seq_num)
    for q in Q:
        print('Now Q is: ')
        print(q)
        for m in M:
            print('Now M is: ', m)
            max_list, min_list = np.zeros(4), np.zeros(4)
            for i in range(seq_num):
                x, y, u = x_list[i], y_list[i], u_list[i]
                print('Sequence Number is: ', i)
                dir_prior = m * q
                # print('Now dir prior is:')
                # print(dir_prior)
                min_error = np.array([1.1] * 4)
                count = 0
                for pi in initial_para:
                    for qi in initial_para:
                        initial_trans = np.array([[pi, 1 - pi], [1 - qi, qi]])
                        # print('Initial count: ', count)
                        y0 = initial_generate(n, start_prob, initial_trans)

                        # Model part
                        y_pred = BayesEM(y0, x, start_prob, dir_prior,
                                         nix_mean, nix_hyper, 1000, 0.01)
                        current_error = np.sum(y != y_pred) / n
                        if current_error < min_error[0]:
                            min_error[0] = current_error
                        # print('BEM error ratio is:', current_error)

                        y_pred = VBEM(y0, x, start_prob, dir_prior,
                                      nix_mean, nix_hyper, 1000, 0.01)
                        current_error = np.sum(y != y_pred) / n
                        if current_error < min_error[1]:
                            min_error[1] = current_error
                        # print('VBEM error ratio is:', current_error)

                        y_pred = SegmentationMM(y0, x, start_prob, dir_prior,
                                                nix_mean, nix_hyper, 1000, 0.01)
                        current_error = np.sum(y != y_pred) / n
                        if current_error < min_error[2]:
                            min_error[2] = current_error
                        # print('SegmentationMM error ratio is:', current_error)

                        y_pred = SegmentationEM(y0, x, start_prob, dir_prior,
                                                nix_mean, nix_hyper, 1000, 0.01)
                        current_error = np.sum(y != y_pred) / n
                        if current_error < min_error[3]:
                            min_error[3] = current_error
                        # print('SegmentationEM error ratio is:', current_error)

                        count += 1
                # print('case ', case, ' and ', q, ' and ', m, ' final result:')
                print('The minimal error is: ', min_error)

                max_val = np.max(min_error)
                min_val = np.min(min_error)
                max_list[min_error == min_val] += 1
                min_list[min_error == max_val] += 1
            print('The maximum list is: ', max_list)
            print('The minimum list is: ', min_list)
