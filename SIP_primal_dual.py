# The case is from Ref. [Mehrotra, 2014]
import numpy as np
import math
import numpy.matlib
import time
import scipy.special
import timeit
# import scipy.stats
import matplotlib.pyplot as plt


class PD:

    def __init__(self, epsilon, num_iterations, num_samples):
        self.num_iterations = num_iterations + 1
        self.num_samples = num_samples  # number of samples of Monte-Carlo simulation
        self.x_dim, self.t_dim = 2, 1  # dimensions of the decision variables / uncertain parameter
        self.arr_x_lb, self.arr_x_ub = np.array([-1, 0]), np.array([1, 0.2])  # [from Ref]
        self.t_lb, self.t_ub = 0, 1
        self.arr_initial_x = np.zeros(self.x_dim)
        self.arr_initial_x = np.array([1, 0.2])
        self.arr_initial_lambda = np.random.uniform(self.t_lb, self.t_ub, self.num_samples)

        # ---- Lip constants and others ---- #
        # self.l_g_eta = np.sqrt(10001)
        self.l_g_eta = 14  # 13.27
        self.l_g_x = np.sqrt(9.5 * 9.5 + 1)
        self.volume_eta = 1  # [from Ref] t \in [0, 1]

        self.epsilon = epsilon
        # ---- Need to calculate this part ---- #
        # self.gamma = 0.1  # didn't calculate yet
        # self.rho_bar = 4  # didn't calculate yet
        # self.rho_0 = self.rho_bar
        # self.mu = 0.5  # didn't calculate yet
        # self.kappa = 0.001  # didn't calculate yet

        self.rho_bar = 7  # 4.2  3.8
        self.rho_0 = self.rho_bar * 1  # rho_0 <= rho_bar
        self.log_r = np.log(np.power(math.pi, self.t_dim / 2) /
                            (scipy.special.gamma(1 + self.t_dim / 2) * self.volume_eta))
        self.H_max = self.rho_0
        self.c_bar = self.rho_bar * (self.l_g_eta * (1 + 1) - self.log_r) + self.H_max
        self.kappa = np.array([self.epsilon / (2 * self.c_bar), (self.epsilon / (2 * self.rho_bar * self.t_dim))**2,
                               self.epsilon / self.rho_0, 1]).min()
        self.c_epsilon_theta = np.maximum(self.rho_bar * (-np.log(self.kappa) * self.t_dim + self.l_g_eta * (1 + 1) -
                                                          self.log_r) + self.H_max, self.rho_0)
        self.diag_x = np.sqrt(4.04)
        self.gamma = 1 * np.sqrt(2 * (self.c_epsilon_theta + self.diag_x) / (num_iterations * (self.rho_bar * 4.75**2
                                                  + 2 * (np.sqrt(2) + self.rho_bar * self.l_g_x)**2)))

        # self.mu = np.sqrt(2 * (self.c_epsilon_theta + self.diag_x) * (num_iterations * (self.rho_bar * 5**2
        #                                           + 2 * (np.sqrt(2) + self.rho_bar * self.l_g_eta)**2)))
        # -------- finished ----------- #
        self.index_g_k = (self.gamma * self.kappa) / (1 + self.gamma * self.kappa)

        self.arr_delta_k = np.zeros([self.num_samples, self.num_iterations])  # array
        self.arr_delta_k[:, 0] = self.arr_initial_lambda
        self.arr_x_k = None
        self.arr_x_bar = None
        self.arr_dual_obj = np.zeros([self.num_iterations, ])
        self.temp = None

        self.fixed_points = None  # fixed sample points in Monte-Carlo simulation
        # -- temp -- #
        self.regenerate_samples = False  # by default is None; If None, plot constraint vio of x_bar
        self.g_max_violation = np.zeros([self.num_iterations, ])

    def grad_x_and_xi(self, x_k_minus):
        # t_samples = np.random.uniform(self.t_lb, self.t_ub, (self.num_samples, ))
        t_samples = self.fixed_points
        grad_x_k = np.zeros([self.x_dim, self.num_samples])
        grad_x_k[0, :] = (10 * np.sin(math.pi * np.sqrt(t_samples)) / (1 + np.power(t_samples, 2))) * x_k_minus[0]
        grad_x_k[1, :] = - np.ones(self.num_samples)
        return grad_x_k

    def generate_monte_carlo_samples(self):
        return np.random.uniform(self.t_lb, self.t_ub, (self.num_samples, ))

    def g_samples(self, x):
        """
        Note that the default order of np.sqrt is 2
        :param x: solution x
        :return: the value of the constraint given the fixed sample points and a solution x
        """
        return (5 * np.sin(math.pi * np.sqrt(self.fixed_points) / (1 + self.fixed_points ** 2))) * x[0] ** 2 - x[1]

    def g_regenerate_samples(self, x):
        if self.regenerate_samples is False:
            samples = self.fixed_points
        else:
            samples = np.random.uniform(self.t_lb, self.t_ub, np.maximum(self.num_samples, 1000))
        return (5 * np.sin(math.pi * np.sqrt(samples) / (1 + samples ** 2))) * x[0] ** 2 - x[1]

    def dual_update(self, x_k_minus, k):
        g_values_temp = self.g_samples(x_k_minus)
        temp_denominator = np.power(self.rho_0 / self.volume_eta, self.index_g_k) * self.volume_eta / self.num_samples \
                           * np.dot(np.exp(g_values_temp * self.gamma / (1 + self.gamma * self.kappa)),
                                    np.power(self.arr_delta_k[:, k - 1], 1 / (1 + self.gamma * self.kappa)))
        temp_outer = np.power(self.rho_0 / self.volume_eta, self.index_g_k) \
                     * np.exp(g_values_temp * self.gamma / (1 + self.gamma * self.kappa)) \
                     * np.power(self.arr_delta_k[:, k - 1], 1 / (1 + self.gamma * self.kappa))
        delta_k = temp_outer if self.rho_bar > temp_denominator \
            else self.rho_bar * temp_outer / temp_denominator
        self.arr_delta_k[:, k] = delta_k
        # calculate the obj of the dual problem
        self.arr_dual_obj[k] = np.sum((x_k_minus - np.array([2, 0.2])) ** 2) \
                               + np.dot(delta_k, g_values_temp.T) / self.num_samples
        self.g_max_violation[k] = np.max(g_values_temp)
        return delta_k

    def sip_primal_dual(self):
        x = np.zeros([self.x_dim, self.num_iterations])
        self.arr_x_bar = np.zeros([self.x_dim, self.num_iterations])
        x[:, 0] = self.arr_initial_x
        self.arr_x_bar[:, 0] = self.arr_initial_x
        self.fixed_points = np.random.uniform(self.t_lb, self.t_ub, self.num_samples)  # Generate fixed sample points
        for k in range(1, self.num_iterations):
            x[:, k] = np.minimum(np.maximum(
                x[:, k - 1] - self.gamma * (2 * (x[:, k - 1] - np.array([2, 0.2])) + self.volume_eta / self.num_samples
                                            * np.dot(self.arr_delta_k[:, k - 1], self.grad_x_and_xi(x[:, k - 1]).T)),
                self.arr_x_lb), self.arr_x_ub)
            self.arr_delta_k[:, k] = self.dual_update(x[:, k - 1], k)
            self.arr_x_bar[:, k] = np.mean(x[:, 0:k], axis=1)
            if self.regenerate_samples is True:
                self.g_max_violation[k] = np.max(self.g_regenerate_samples(x[:, k]))
        x_bar = np.mean(x, axis=1)
        f_x = (x_bar[0] - 2)**2 + (x_bar[1] - 0.2)**2
        self.arr_x_k = x

        constraint_violation = np.max(self.g_regenerate_samples(x_bar))

        print('f(x_K) is', np.sum((x[:, self.num_iterations - 1] - np.array([2, 0.2])) ** 2), 'f(x_bar_K) is ', f_x,
              ' The optimum is', np.sum((np.array([[0.20523677], [0.2]]) - np.array([[2], [0.2]])) ** 2),
              ' Constraint violation is', constraint_violation)

        return f_x, x_bar, constraint_violation

    def plot_result(self, normal_plot=True):
        if normal_plot is True:
            plt.plot(range(self.num_iterations), np.sum((self.arr_x_k - np.array([[2], [0.2]])) ** 2, axis=0),
                     marker='x', markersize=1, color='b', linewidth=0.5)  # plot the obj of primal
            # plt.plot(range(self.num_iterations), np.sum((self.arr_x_bar - np.array([[2], [0.2]])) ** 2, axis=0),
            #          marker='o', markersize=1, color='r', linewidth=0.5)  # plot the obj of primal
            plt.plot(range(self.num_iterations),
                     np.sum((np.array([[0.20523677], [0.2]]) - np.array([[2], [0.2]])) ** 2) * np.ones(
                         self.num_iterations), linestyle='--', color='k', linewidth=0.5)  # plot the obj of primal
            plt.legend([r'$f(x_k)$', 'Optimal value'], fontsize=10.5)
        else:  # plot in log x-axis
            plt.semilogx(range(self.num_iterations), np.sum((self.arr_x_k - np.array([[2], [0.2]])) ** 2, axis=0),
                         marker='x', markersize=1, color='b', linewidth=0.5)  # plot the obj of primal
            plt.semilogx(range(self.num_iterations),
                         np.sum((np.array([[0.20523677], [0.2]]) - np.array([[2], [0.2]])) ** 2) * np.ones(
                         self.num_iterations), linestyle='--', color='k', linewidth=0.5)  # plot the obj of primal
            plt.legend([r'$f(x_k)$', 'Optimal value'], fontsize=10.5)

        # plt.plot(range(self.num_iterations), self.arr_dual_obj, linestyle='--', color='k', linewidth=1)

    def plot_constraint_violation(self, normal_plot=True):
        if normal_plot is True:
            plt.plot(range(self.num_iterations), self.g_max_violation[0:self.num_iterations],
                     marker='x', markersize=1, color='b', linewidth=0.5)
        else:
            plt.semilogx(range(self.num_iterations), self.g_max_violation[0:self.num_iterations],
                         marker='x', markersize=1, color='b', linewidth=0.5)

    def calculate_constraint_violation_x_bar(self):

        pass


def sensitivity_analysis_on_k():
    k_candidate = [500, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
    # k_candidate = [1000, 3000, 5000, 10000]
    # k_candidate = [1000, 5000, 10000]
    # k_candidate = [500, 1000, 1500]

    f_x_record = np.zeros(len(k_candidate))
    con_vio_record = np.zeros(len(k_candidate))
    i = 0
    for k_num in k_candidate:
        pd_algorithm = PD(0.001, k_num, 1000)
        f_x_record[i], x_bar, con_vio_record[i] = pd_algorithm.sip_primal_dual()
        i += 1
    plt.plot(k_candidate, f_x_record,
             marker='x', markersize=3, color='b', linewidth=1)  # plot the obj of primal
    plt.plot(range(max(k_candidate)),
             np.sum((np.array([[0.20523677], [0.2]]) - np.array([[2], [0.2]])) ** 2) * np.ones(
                 max(k_candidate)), linestyle='--', color='k', linewidth=1)
    plt.legend([r'$f(\bar{x}_K)$', 'Optimal value'], fontsize=10.5)
    plt.xlabel(r'$K$', fontsize=10.5)
    plt.ylabel(r'Objective Value $f(\bar{x}_K)$', fontsize=10.5)
    plt.show()

    plt.plot(k_candidate, con_vio_record, marker='x', markersize=3, color='b', linewidth=1)  # constraint violation
    plt.plot(range(max(k_candidate)), np.zeros(max(k_candidate)),
             linestyle='--', color='k', linewidth=0.5)
    plt.legend([r'$G(\bar{x}_K$)', 'Constraint violation'], fontsize=10.5)
    plt.xlabel(r'$K$', fontsize=10.5)
    plt.ylabel(r'G($\bar{x}_K$)', fontsize=10.5)
    plt.show()

    pd_algorithm.plot_result()
    plt.show()

    pd_algorithm.plot_result(False)
    plt.show()


def sensitivity_anlysis_on_sampling():
    n_candidate = [10, 100, 1000]
    k_iteration = 50000
    lst_color = ['b', 'r', 'g', 'p']
    lst_marker = ['x', 'o', '*']

    f_x_record = np.zeros(len(n_candidate))
    lst_con_vio_save = [None] * len(n_candidate)
    i = 0
    for n_num in n_candidate:
        pd_algorithm = PD(0.001, k_iteration, n_num)
        pd_algorithm.regenerate_samples = True  # if it is on, regenerate samples to evaluate constraint violation
        f_x_record[i], x_bar, temp = pd_algorithm.sip_primal_dual()
        plt.semilogx(range(pd_algorithm.num_iterations), np.sum((pd_algorithm.arr_x_k - np.array([[2], [0.2]])) ** 2,
                                                                axis=0),
                     marker=lst_marker[i], markersize=1, color=lst_color[i], linewidth=0.5)  # plot the obj of primal
        lst_con_vio_save[i] = pd_algorithm.g_max_violation[0:pd_algorithm.num_iterations].copy()
        i += 1

    plt.plot(range(k_iteration),
             np.sum((np.array([[0.20523677], [0.2]]) - np.array([[2], [0.2]])) ** 2) * np.ones(
                 k_iteration), linestyle='--', color='k', linewidth=1)
    plt.legend([r'$N=10$', r'$N=100$', r'$N=1000$', 'Optimal value'], fontsize=10.5)
    plt.xlabel(r'$k$', fontsize=10.5)
    plt.ylabel(r'Objective Value $f(x_k)$', fontsize=10.5)
    plt.show()

    for j in range(len(n_candidate)):
        plt.semilogx(range(k_iteration), lst_con_vio_save[j][1:k_iteration + 1], marker=lst_marker[j], markersize=1,
                     color=lst_color[j], linewidth=0.5)
    plt.plot(range(k_iteration), np.zeros(k_iteration),
             linestyle='--', color='k', linewidth=0.5)
    plt.legend([r'$N=10$', r'$N=100$', r'$N=1000$', 'Constraint violation'], fontsize=10.5)
    plt.xlabel(r'$k$', fontsize=10.5)
    plt.ylabel(r'G($x_k$)', fontsize=10.5)
    plt.show()


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    np.random.seed(4)
    # pd_algorithm = PD(0.001, 1000, 1000)
    # normal_plot = False  # if yes, x-axis is normal, otherwise is log x
    # start_time = time.time()
    # f_x, x_bar = pd_algorithm.sip_primal_dual()
    # print('run time is:', time.time() - start_time)
    # # pd_algorithm.plot_result(normal_plot)
    # # plt.show()
    # pd_algorithm.plot_constraint_violation(normal_plot)
    # plt.show()
    # print('f(x_bar) is:', f_x)
    # print('x_bar is:', x_bar)

    # sensitivity_analysis_on_k()

    sensitivity_anlysis_on_sampling()


if __name__ == '__main__':
    main()
