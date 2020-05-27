# SIP practice: problem HET_Z
import numpy as np
import math
import numpy.matlib
import time
import timeit
# import scipy.stats
import matplotlib.pyplot as plt


class CSA:

    def __init__(self, parse_input):
        self.epsilon = parse_input['epsilon']
        self.num_iterations = parse_input['num_iterations']
        self.x0 = parse_input['x0']

        self.c_gamma, self.c_eta = parse_input['c_gamma'], parse_input['c_eta']  # to adjust gamma/eta in simulations

        self.x_dim = len(self.x0)  # dimension of x
        self.delta_dim = 1

        # ========= objective function ========= #
        self.dim_num_s = self.x_dim
        self.grad_objective = np.array([0, 1])  # gradients of the obj function

        # ========= decision variables ========= #
        self.x_ub = 2  # np.inf
        self.x_lb = - 2  # -np.inf

        # ========= Delta region ========= #
        self.delta_lb = -1
        self.delta_ub = 1

        # ========= calculate the parameters ========= #
        self.l_g_x = 2  # given any delta, L-constant of g
        self.l_f = 1  #
        self.l_g_delta = max(2 * self.x_ub,  1)  # ## given any x, L-constant of g(x, delta)
        self.d_x = (self.x_ub - self.x_lb) * np.sqrt(self.x_dim)  # diameter of the domain x
        self.d_delta = self.delta_ub - self.delta_lb  # diameter of delta
        self.r_delta = (self.delta_ub - self.delta_lb) / 2  # radius of the largest ball which can be included in Delta
        self.parameter_c = self.l_g_delta * (self.r_delta + self.d_delta) - np.log(1)

        # output setup #
        # self.output_x_sol = False  # if it is true, output itertative obj; otherwise output obj_bar
        self.plot_len = self.num_iterations
        self.plot_linecolor, self.plot_linestyle, self.plot_linewid = None, None, 0.9
        # self.plot_constraint_violation = None  # if true, plot constraint violations
        self.accumulate_num_of_bb = np.zeros(self.num_iterations + 1)
        self.obj_weighted_sum = np.zeros([self.num_iterations + 1])
        self.x_sol = np.zeros([self.x_dim, self.num_iterations + 1])

        # -- some attributes for plot
        self.arr_x_bar_with_k = np.zeros([self.x_dim, self.num_iterations + 1])

    def mapping_to_feasible_region(self, x):
        return np.maximum(np.minimum(x, self.x_ub), self.x_lb)

    def fixed_constraints_sampling(self, x):
        num_samples = self.fixed_num_samples
        random_samples = np.random.uniform(self.delta_lb, self.delta_ub, num_samples)

        arr_g_grad, arr_g_values = self.calculate_g_grad_and_values_arr(x, random_samples)

        max_id = np.where(arr_g_values == arr_g_values.max())
        return arr_g_grad[:, max_id[0][0]], arr_g_values[max_id[0][0]]

    def adaptive_constraint_sampling(self, x, k=0, iteration=800):
        selected_samples = self.adaptive_selected_samples  # M; default is 1
        epsilon_k = (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
        # epsilon_k = 1 / np.sqrt(k + 1)
        kappa = np.minimum(np.minimum(epsilon_k / (2 * self.parameter_c), (epsilon_k / (2 * self.delta_dim))**2), 1)
        theta_sample = np.zeros([iteration])
        theta_rand = np.random.uniform(self.delta_lb, self.delta_ub, iteration)
        theta_sample[0] = theta_rand[0].copy()
        t = 1
        while t < iteration:
            _, g_value = self.calculate_g_grad_and_values(x, theta_rand[t])
            alpha = np.minimum(1, np.exp((
                                             g_value  # problem specific
                                         ) / kappa))
            u = np.random.uniform(0, 1)
            sign_u = np.sign(np.sign(u - alpha) + 1)
            theta_sample[t] = (theta_rand[t].T * (1 - sign_u) + theta_sample[t - 1].T * sign_u).T
            t += 1
        temp_values, arr_g_grad_max = np.zeros([selected_samples + 1]), np.zeros([self.x_dim])
        temp_values[0] = -10e9
        temp0 = -10e9
        if selected_samples == 1:
            arr_g_grad, temp_values[1] = self.calculate_g_grad_and_values(x, theta_rand[-1])
            if temp_values[1].max() > temp0:
                arr_g_grad_max = arr_g_grad
                temp0 = temp_values[1]
        else:
            arr_g_grad, temp_values = self.calculate_g_grad_and_values_arr(x, theta_rand[iteration -
                                                                                         selected_samples:iteration])
            max_id = np.where(temp_values == temp_values.max())
            arr_g_grad_max, temp0 = arr_g_grad[:, max_id[0][0]], temp_values[max_id[0][0]]
        return arr_g_grad_max, temp0

    def calculate_g_grad_and_values_arr(self, x, theta_rand_value):
        values_inside_abs = 1 - np.power(theta_rand_value, 2) - (0.5 * x[0] ** 2 - 2 * x[0] * theta_rand_value)  # values in abs
        pos_id = np.where(values_inside_abs >= 0)[0]
        neg_id = np.setdiff1d(range(len(values_inside_abs)), pos_id, assume_unique=True)
        g_grad = np.zeros([self.x_dim, len(values_inside_abs)])
        if len(pos_id) > 0:
            g_grad[:, pos_id] = np.array([-np.ones(len(pos_id)) * x[0] + 2 * theta_rand_value[pos_id],
                                          -np.ones(len(pos_id))])  # gradients
        if len(neg_id) > 0:
            g_grad[:, neg_id] = np.array([np.ones(len(neg_id)) * x[0] - 2 * theta_rand_value[neg_id],
                                          -np.ones(len(neg_id))])  # gradients
        g_values = np.abs(values_inside_abs) - x[-1]
        return g_grad, g_values

    def calculate_g_grad_and_values(self, x, theta_rand_value):
        values_inside_abs = 1 - np.power(theta_rand_value, 2) - (0.5 * x[0] ** 2 - 2 * x[0] * theta_rand_value)   # values in abs
        g_grad = np.zeros([self.x_dim])
        g_grad = np.array([-x[0] + 2 * theta_rand_value, -1]) if values_inside_abs >= 0 else \
            np.array([x[0] - 2 * theta_rand_value, -1])
        g_values = np.abs(values_inside_abs) - x[-1]
        return g_grad, g_values

    def csa_algorithm(self):
        num_iteration = self.num_iterations
        # initial value of x_sol
        self.x_sol[:, 0] = self.x0
        if self.x_sol[:, 0].max() > self.x_ub or self.x_sol[:, 0].min() < self.x_lb:
            print('The initial value of x is invalid!')

        gamma = np.zeros(num_iteration + 1)
        eta = np.zeros(num_iteration + 1)
        count = 0
        index_set = []
        for k in range(num_iteration):
            gamma[k] = self.c_gamma * self.d_x / (np.sqrt(k + 1) * (self.l_f + self.l_g_x))
            eta[k] = self.c_eta * 6 * (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)

            grad_a_k, value_a_k = self.fixed_constraints_sampling(self.x_sol[:, k]) if self.fixed_sampling is True \
                else self.adaptive_constraint_sampling(self.x_sol[:, k], k, self.adaptive_iterations)
            if value_a_k <= eta[k]:
                vec_h_k = self.grad_objective
                index_set.append(k)
            else:
                vec_h_k = grad_a_k
            self.x_sol[:, k + 1] = self.mapping_to_feasible_region(self.x_sol[:, k] - 0.5 * gamma[k] * vec_h_k)  # L2-norm

            # to calculate x_bar
            idx_s = list(set(index_set) & set(range(int(np.ceil(k / 2)), k + 1)))
            if len(idx_s) > 0:
                self.obj_weighted_sum[k] = np.dot(gamma[idx_s], np.dot(self.grad_objective, self.x_sol[:, idx_s])) \
                                           / np.sum(gamma[idx_s])
                self.arr_x_bar_with_k[:, k] = np.sum((np.matlib.repmat(gamma[idx_s], self.x_dim, 1) * self.x_sol[:, idx_s] /
                                                      np.sum(gamma[idx_s])), axis=1)
            else:
                self.obj_weighted_sum[k] = float('inf')
                print('Empty set_B is found in iteration:', k)

            count += 1
            if count == 2000:
                aa = 1

            # if self.plot_setB is True:
            self.accumulate_num_of_bb[k] = len(index_set)

        x_sum = np.zeros(self.x_dim)
        gamma_sum = 0
        # subset = set(index_set) & set(range(num_iteration - 5000, num_iteration + 1, 1))
        subset = set(index_set) - set(range(int(np.ceil(num_iteration / 2) - 1)))
        for j in subset:
            x_sum += gamma[j] * self.x_sol[:, j]
            gamma_sum += gamma[j]
        x_bar = x_sum / gamma_sum
        obj = np.dot(self.grad_objective, x_bar)
        return x_bar, obj

    def plot_constraint_violation(self, out_of_samples=10000):
        self.fixed_num_samples = out_of_samples
        arr_violation = np.zeros(self.num_iterations + 1)
        for k in range(self.num_iterations + 1):
            grad_a_k, value_a_k = self.fixed_constraints_sampling(self.arr_x_bar_with_k[:, k])
            arr_violation[k] = value_a_k
        plt.semilogx(range(self.plot_len), arr_violation[0:self.plot_len],
                     color=self.plot_linecolor, linestyle=self.plot_linestyle, linewidth=self.plot_linewid)

    def plot_set_bb(self):
        plt.plot(range(self.plot_len), self.accumulate_num_of_bb[0:self.plot_len])
        return

    def plot_x_last_iterate(self):
        plt.semilogx(range(self.plot_len), np.dot(self.grad_objective, self.x_sol[:, 0:self.plot_len]),
                     color=self.plot_linecolor, linestyle=self.plot_linestyle, linewidth=self.plot_linewid)

    def plot_x_bar(self):
        plt.semilogx(range(self.plot_len), self.obj_weighted_sum[0:self.plot_len],
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
    parse_input = {
        'epsilon': 0.01,
        'num_iterations': 1000,
        'c_gamma': 0.05,
        'c_eta': 0.001,
        'x0': np.array([1, 1])  # according to the paper, the third one is eta, whose maximum is 5.389 if x=(1,1)
    }
    csa = CSA(parse_input)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    csa.run_adaptive_sampling(10, 50)
    # csa.run_fixed_sampling(1000)
    csa.plot_x_last_iterate()
    csa.plot_x_bar()
    plt.show()


if __name__ == '__main__':
    main()
