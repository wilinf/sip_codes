import matplotlib.pyplot as plt
import numpy as np

from SIP_csa_algorithm import CSA

# font = {'family': 'Times New Roman',
#         'weight': 'light',
#         'size': 10.5,
#         }

np.random.seed(1)
parse_input = {
    'epsilon': 0.001,
    'num_iterations': 1000,
    'c_gamma': 0.35,
    'c_eta': 0.001,
    'x0': np.zeros(2)
}
csa = CSA(parse_input)


def run_simulation_objective_and_violations():
    lst_arr_x_bar_with_k = []
    lst_obj_weighted_sum = []
    csa.run_adaptive_sampling(50, 50)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
    lst_obj_weighted_sum.append(csa.obj_weighted_sum.copy())
    fixed_sampling_candidates = [10, 20, 50, 100,]
    for sample_size in fixed_sampling_candidates:
        csa.run_fixed_sampling(sample_size)
        lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
        lst_obj_weighted_sum.append(csa.obj_weighted_sum.copy())

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(len(lst_obj_weighted_sum)):
        csa.plot_x_bar(lst_obj_weighted_sum[i])
    plt.plot(range(parse_input['num_iterations']), -1.559 * np.ones(parse_input['num_iterations']), linestyle='--',
             color='k', linewidth=1)
    sampling_legend = ['Adaptive sampling']
    sampling_legend.extend(['Fixed sampling - {}'.format(i) for i in fixed_sampling_candidates] + ['Optimal value'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iterations')
    plt.legend(sampling_legend, fontsize=10.5)
    plt.ylim(-1.8, -0.7)
    plt.show()

    plt.close()
    for i in range(len(lst_arr_x_bar_with_k)):
        csa.plot_constraint_violation(lst_arr_x_bar_with_k[i])
    plt.plot(range(csa.plot_len), 0 * np.ones(csa.plot_len), linestyle='--', color='k', linewidth=1)
    sampling_legend = ['Adaptive sampling']
    sampling_legend.extend(['Fixed sampling - {}'.format(i) for i in fixed_sampling_candidates] +
                           ['Constraint Violation'])
    plt.ylabel(r'$G(\bar{x}_{N,s})$', fontsize=10.5)
    plt.xlabel('Iterations')
    plt.legend(sampling_legend, fontsize=10.5)
    # plt.ylim(-1.8, -0.7)
    plt.show()
    return


def plot_set_b():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    csa.run_adaptive_sampling(50, 50)
    csa.plot_set_bb()
    csa.run_fixed_sampling(500)
    csa.plot_set_bb()

    n_sequence = np.arange(0, 1000)
    plt.plot(n_sequence, (n_sequence - np.ceil(n_sequence/2) + 1)/2, linestyle='--', color='k', linewidth=1)
    plt.legend(['Adaptive sampling', 'Fixed sampling - 100', r'(N-s+1)/2'], fontsize=10.5)
    plt.xlabel('Iterations',)
    plt.ylabel('$\| \mathcal{B} \|$', fontsize=10.5)
    plt.show()
    return


def sensitivity_adaptive_sampling():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    candidate_num = [5 * (i + 1) for i in range(10)]
    candidate_num = [5, 7, 10, 15, 20, 30, 40, 50, 60]
    obj = np.zeros(candidate_num.__len__())
    i = 0
    for num_iter_mcmc in candidate_num:
        a, obj[i] = csa.run_adaptive_sampling(num_iter_mcmc, num_iter_mcmc)
        i += 1
    plt.plot(candidate_num, obj, marker='x', color='b', linewidth=1)
    plt.plot(range(max(candidate_num)), -1.559 * np.ones(max(candidate_num)), linestyle='--', color='k', linewidth=1)
    plt.xlabel('Iterations for the MH algorithm', fontsize=10.5)
    plt.ylabel('Objective Value', fontsize=10.5)
    plt.show()
    return


if __name__ == '__main__':
    run_simulation_objective_and_violations()
    # plot_set_b()
    # sensitivity_adaptive_sampling()
    pass
