# SIP practice
import numpy as np
import math
import numpy.matlib
import time
import timeit
# import scipy.stats
import matplotlib.pyplot as plt


class CSA:

    def __init__(self, eta, epsilon, num_iterations):
        self.beta, self.epsilon, self.num_iterations = eta, epsilon, num_iterations
        self.constraint_norm = 1
        self.cost_objective = np.array([-1, -1])
        self.mat_a = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]) * self.constraint_norm
        self.coeff = 0.2 * self.constraint_norm
        self.bb = np.array([0, 0, 1, 1])
        self.x_dim, self.delta_dim = 2, 8
        temp = np.sqrt(36/37) + 0.2 * (np.sqrt(36/37) + np.sqrt(1/37))
        temp = 1.2
        self.l_g_x, self.l_f = temp, np.sqrt(2)  # np.sqrt(2)
        self.l_g_delta = 2 * 0.2
        self.x_ub, self.x_lb = 2, -2
        self.d_x = (self.x_ub - self.x_lb) * np.sqrt(2)
        self.d_delta = 4
        self.r_delta = 1
        self.c_gamma, self.c_eta = 1, 1  # to adjust gamma/eta in simulations
        self.parameter_c = self.l_g_delta * (self.r_delta + self.d_delta) - np.log(1)
        self.fixed_sampling = True
        self.adaptive_iterations = 800
        self.adaptive_selected_samples = 1
        self.fixed_num_samples = 3000
        self.output_x_sol = False  # if it is true, output itertative obj; otherwise output obj_bar
        self.plot_setB = False  # if it is true, plot the change of set B (|B|) with the increase of iterations
        self.plot_len = None
        self.plot_linecolor, self.plot_linestyle, self.plot_linewid = None, None, 0.9
        self.compare = True
        self.plot_constraint_violation = None  # if true, plot constraint violations
        self.initial_x = np.zeros(2)
        # -- some attributes for plot
        self.arr_x_bar_with_k = np.zeros([self.x_dim, self.num_iterations + 1])

    def fixed_constraints_sampling(self, x):
        # num_samples = int(np.ceil(np.log(self.beta)/np.log(1 - self.epsilon)))

        # epsilon_k = (self.l_g_x + self.l_f) * 2 * np.sqrt(2) / np.sqrt(k + 1)
        # num_samples = int(np.ceil(np.log(np.maximum(np.minimum(epsilon_k, 1), 0))
        #                       / np.log(1 - epsilon_k / (2 * (10000 - 1000)))))

        # num_samples = np.maximum(np.minimum(num_samples, 3000), 1000)
        num_samples = self.fixed_num_samples

        random_samples = self.uniform_ball(1, 4 * num_samples)
        x_adjust = np.zeros([4, x.shape[0]])
        for i in range(4):
            value_temp = np.dot(x, np.matlib.repmat(self.mat_a[i, :].T, num_samples, 1).T
                                + self.coeff * random_samples[:, i * num_samples:(i + 1) * num_samples])
            loc = np.where(value_temp == value_temp.max())
            x_adjust[i, :] = random_samples[:, i * num_samples:(i + 1) * num_samples][:, loc[0][0]]
        mat_big_a = self.mat_a + self.coeff * x_adjust
        mat_a_values = np.dot(mat_big_a, x.T) - self.bb
        max_id = np.where(mat_a_values == mat_a_values.max())
        return mat_big_a[max_id[0][0], :], mat_a_values[max_id[0][0]]

    def adaptive_constraint_sampling(self, x, k=0, iteration=800):
        selected_samples = self.adaptive_selected_samples  # M; default is 1
        epsilon_k = (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
        # epsilon_k = 1 / np.sqrt(k + 1)
        kappa = np.minimum(np.minimum(epsilon_k / (2 * self.parameter_c), (epsilon_k / (2 * self.delta_dim))**2), 1)
        theta_rand = np.zeros([4, 2, iteration])
        theta_sample = np.zeros([4, 2, iteration])
        for i in range(4):
            theta_rand[i, :, :] = self.uniform_ball(1, iteration)
        theta_sample[:, :, 0] = theta_rand[:, :, 0].copy()
        t = 1
        while t < iteration:
            alpha = np.minimum(1, np.exp((np.dot(self.coeff * (theta_rand[:, :, t]
                                                        - theta_sample[:, :, t - 1]), x.T)) / kappa))
            u = np.random.uniform(0, 1, (4, ))
            sign_u = np.sign(np.sign(u - alpha) + 1)
            theta_sample[:, :, t] = (theta_rand[:, :, t].T * (1 - sign_u) + theta_sample[:, :, t - 1].T * sign_u).T
            t += 1
        temp_values, mat_big_a = np.zeros([4, selected_samples + 1]), np.zeros([4, 2])
        temp_values[:, 0], max_id = -10e9 * np.ones(4), 0
        temp0 = -10e9
        for i in range(selected_samples):
            temp_values[:, i + 1] = np.dot(self.mat_a + self.coeff * theta_sample[:, :, iteration - i - 1], x.T) - self.bb
            if temp_values[:, i + 1].max() > temp0:
                max_id = np.where(temp_values[:, i + 1] == temp_values[:, i + 1].max())[0][0]
                mat_big_a = self.mat_a[max_id, :] + self.coeff * theta_sample[max_id, :, iteration - i - 1]
                temp0 = temp_values[:, i + 1].max()
        return mat_big_a, temp0

    def csa_algorithm(self):
        num_iteration = self.num_iterations
        x_sol = np.zeros([self.x_dim, num_iteration + 1])
        # initial value of x_sol
        x_sol[:, 0] = self.initial_x
        if x_sol[:, 0].max() > self.x_ub or x_sol[:, 0].min() < self.x_lb:
            print('The initial value of x is invalid!')
        obj_weighted_sum = np.zeros([num_iteration + 1])
        gamma = np.zeros(num_iteration + 1)
        eta = np.zeros(num_iteration + 1)
        accumulate_num_of_B = np.zeros(num_iteration + 1)
        index_set = []
        for k in range(num_iteration):
            norm_x = np.linalg.norm(x_sol[:, k])
            # d_x_k = np.sqrt(2 * np.sqrt(2) + np.dot(np.abs(x_sol[:, k]), np.array([4, 4])) / norm_x) \
            #     if norm_x > 0 else np.sqrt(2 * np.sqrt(2))
            gamma[k] = self.c_gamma * self.d_x / (np.sqrt(k + 1) * (self.l_f + self.l_g_x))
            # gamma[k] = self.d_x * 0.2 / (np.sqrt(k + 1) * (self.l_f + self.l_g_x))
            # gamma[k] = self.d_x * 0.3 / (np.sqrt(k + 1) * (self.l_f + self.l_g_x))
            # gamma[k] = 1 / (np.sqrt(k + 1) * (self.l_f + self.l_g_x))
            # eta[k] = 6 * (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
            eta[k] = self.c_eta * 6 * (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
            # eta[k] = 0.2 * (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
            # eta[k] = 1 / np.sqrt(k + 1)
            # eta[k] = 1.2 / np.sqrt(k + 1)  # eta 2
            # eta[k] = 0.2 * (2 + 0.2 * (2 * np.sqrt(2))) / np.sqrt(k + 1)  # eta 3
            # eta[k] = (2 + 0.2 * (2 * np.sqrt(2))) / np.sqrt(k + 1)  # eta 1
            # eta[k] = 0
            # eta[k] = 1.2 / np.sqrt(k + 1) if self.compare is True \
            #     else 0.2 * (2 + 0.2 * (2 * np.sqrt(2))) / np.sqrt(k + 1)

            grad_a_k, value_a_k = self.fixed_constraints_sampling(x_sol[:, k]) if self.fixed_sampling is True \
                else self.adaptive_constraint_sampling(x_sol[:, k], k, self.adaptive_iterations)
            if value_a_k <= eta[k]:
                vec_h_k = self.cost_objective
                index_set.append(k)
            else:
                vec_h_k = grad_a_k
            x_sol[:, k + 1] = np.maximum(np.minimum(x_sol[:, k] - 0.5 * gamma[k] * vec_h_k, self.x_ub), self.x_lb)
            if self.output_x_sol is False:
                # to calculate x_bar
                idx_s = list(set(index_set) & set(range(int(np.ceil(k / 2)), k + 1)))
                if len(idx_s) > 0:
                    obj_weighted_sum[k] = np.dot(gamma[idx_s], -np.sum(x_sol[:, idx_s], axis=0)) \
                                               / np.sum(gamma[idx_s])
                    self.arr_x_bar_with_k[:, k] = np.sum((np.matlib.repmat(gamma[idx_s], self.x_dim, 1) \
                                                  * x_sol[:, idx_s] / np.sum(gamma[idx_s])), axis=1)
                else:
                    obj_weighted_sum[k] = float('inf')
                    print('Empty set_B is found in iteration:', k)

            if self.plot_setB is True:
                accumulate_num_of_B[k] = len(index_set)

        x_sum = np.zeros(2)
        gamma_sum = 0
        # subset = set(index_set) & set(range(num_iteration - 5000, num_iteration + 1, 1))
        subset = set(index_set) - set(range(int(np.ceil(num_iteration / 2) - 1)))
        for j in subset:
            x_sum += gamma[j] * x_sol[:, j]
            gamma_sum += gamma[j]
        x_bar = x_sum / gamma_sum
        obj = np.dot(np.array([-1, -1]), x_bar)
        if self.plot_setB is True and self.plot_constraint_violation is False:
            plt.plot(range(self.plot_len), accumulate_num_of_B[0:self.plot_len])
        elif self.plot_constraint_violation is False:
            plt.semilogx(range(self.plot_len), -np.sum(x_sol[:, 0:self.plot_len], axis=0),
                         color=self.plot_linecolor, linestyle=self.plot_linestyle, linewidth=self.plot_linewid) \
                if self.output_x_sol is True \
                else plt.semilogx(range(self.plot_len), obj_weighted_sum[0:self.plot_len],
                                  color=self.plot_linecolor, linestyle=self.plot_linestyle, linewidth=self.plot_linewid)
            # plt.plot(range(self.plot_len), -np.sum(x_sol[:, 0:self.plot_len], axis=0)) if self.output_x_sol is True \
            #     else plt.plot(range(self.plot_len), obj_weighted_sum[0:self.plot_len])
        # if self.plot_constraint_violation is True:
            # self._plot_constraint_violation()
        return x_bar, obj

    def _plot_constraint_violation(self, out_of_samples=10000):
        self.fixed_num_samples = out_of_samples
        arr_violation = np.zeros(self.num_iterations + 1)
        for k in range(self.num_iterations + 1):
            grad_a_k, value_a_k = self.fixed_constraints_sampling(self.arr_x_bar_with_k[:, k])
            arr_violation[k] = value_a_k
        plt.semilogx(range(self.plot_len), arr_violation[0:self.plot_len],
                     color=self.plot_linecolor, linestyle=self.plot_linestyle, linewidth=self.plot_linewid)

    def run_fixed_sampling(self, fixed_num_samples=1000):
        self.fixed_sampling = True
        self.fixed_num_samples = fixed_num_samples
        start_time = time.time()
        x_opt, obj = self.csa_algorithm()
        print('Obj is', obj, ' Optimal sol is', x_opt)
        print('Run time is:', time.time() - start_time)
        return x_opt, obj

    def run_adaptive_sampling(self, adaptive_selected_samples=1, adaptive_iterations=200):
        # adaptive_selected_samples: how many samples selected after MCMC is run
        # adaptive_iterations: # of iterations for MCMC
        self.fixed_sampling = False
        self.adaptive_selected_samples = adaptive_selected_samples
        self.adaptive_iterations = adaptive_iterations
        start_time = time.time()
        x_opt, obj = self.csa_algorithm()
        print('Obj is', obj, ' Optimal sol is', x_opt)
        print('Run time is:', time.time() - start_time)
        return x_opt, obj


    @staticmethod
    def uniform_ball(r=1, num_samples=1000):
        x = np.zeros([2, num_samples])
        u = r * np.sqrt(np.random.uniform(0, 1, num_samples))
        theta = np.random.uniform(0, 1, num_samples) * 2 * math.pi
        x[0, :] = u * np.sin(theta)
        x[1, :] = u * np.cos(theta)
        return x

    @staticmethod
    def other_method(dim=2):
        size = 1000
        x = np.random.normal(size=(size, dim))
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
        return x

    @staticmethod
    def _runtime_test():
        start_time = time.time()
        for i in range(int(10e3)):
            # put your test codes here

            pass
        print('run time is:', time.time() - start_time)


def main():
    np.random.seed(1)
    csa = CSA(0.001, 0.001, 1000)
    csa.plot_len = csa.num_iterations
    csa.output_x_sol = False
    csa.plot_setB = False
    csa.plot_constraint_violation = True
    csa.initial_x = np.array([0, 0])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    csa.c_gamma, csa.c_eta = 0.35, 0.001
    # csa.c_gamma, csa.c_eta = 1, 0.11
    # csa.c_eta = 1.2 * (csa.x_ub + csa.x_ub * np.sqrt(2) / 5) / (6 * (csa.l_f + csa.l_g_x) * csa.d_x)
    # csa.c_gamma = csa.c_eta
    lst_arr_x_bar_with_k = []
    csa.run_adaptive_sampling(10, 50)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
    csa.run_fixed_sampling(10)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
    csa.run_fixed_sampling(20)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
    csa.run_fixed_sampling(50)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())
    csa.run_fixed_sampling(100)
    lst_arr_x_bar_with_k.append(csa.arr_x_bar_with_k.copy())

    for i in range(5):
        csa.arr_x_bar_with_k = lst_arr_x_bar_with_k[i]
        csa._plot_constraint_violation()

    # candidate_num = [10, 20, 50, 100, 150, 200]
    # obj = np.zeros(candidate_num.__len__())
    # i = 0
    # for num_iter_mcmc in candidate_num:
    #     a, obj[i] = csa.run_adaptive_sampling(1, num_iter_mcmc)
    #     i += 1
    # plt.show()
    # plt.plot(candidate_num, obj, marker='x', color='b', linewidth=1)
    # plt.plot(range(max(candidate_num)), -1.559 * np.ones(max(candidate_num)), linestyle='--', color='k', linewidth=1)

    if csa.plot_setB is False:
        # plt.semilogx(range(csa.plot_len), -1.559 * np.ones(csa.plot_len), linestyle='--', color='k', linewidth=1)
        plt.semilogx(range(csa.plot_len), 0 * np.ones(csa.plot_len), linestyle='--', color='k', linewidth=1)
        # plt.plot(range(csa.plot_len), -1.559 * np.ones(csa.plot_len), linestyle='--', color='k', linewidth=1)
        # plt.legend(['Fixed sampling',  'Optimal value', 'Adaptive sampling'])
    else:
        n_sequence = np.arange(0, 1000)
        plt.plot(n_sequence, (n_sequence - np.ceil(n_sequence/2) + 1)/2, linestyle='--', color='k', linewidth=1)
        plt.legend(['Adaptive sampling', 'Fixed sampling - 100', r'(N-s+1)/2'], fontsize=10.5)
        # plt.ylabel('Objective Value')
        # plt.xlabel('Iterations')

    # for fixed_num_samples in [10, 20, 50, 100]:
    #     csa.run_fixed_sampling(fixed_num_samples)
    # plt.semilogx(range(csa.plot_len), -1.559 * np.ones(csa.plot_len), linestyle='--', color='k', linewidth=1)
    # plt.legend(['Adaptive sampling', 'Fixed sampling - 10', 'Fixed sampling - 20',
    #             'Fixed sampling - 50', 'Fixed sampling - 100'], fontsize=10.5)
    plt.legend(['Adaptive sampling', 'Fixed sampling - 10', 'Fixed sampling - 20',
                'Fixed sampling - 50', 'Fixed sampling - 100', 'Constraint Violation'], fontsize=10.5)
    # plt.ylabel('Unviolated Constraints')
    # plt.ylabel('Objective Value', fontsize=10.5)
    plt.xlabel('Iterations', fontsize=10.5)
    # plt.xlabel('Iterations for MCMC', fontsize=10.5)
    # plt.ylabel(r'G($\bar{x}_{N,s}$)', fontsize=10.5)

    plt.show()


if __name__ == '__main__':
    main()