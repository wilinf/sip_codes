# SIP practice: problem anderson1
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
        self.grad_objective = np.array([1/i for i in range(1, self.dim_num_s + 1)])  # gradients of the obj function

        # ========= constraints ========= #
        self.mat_a = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        self.coeff = 0.2
        self.bb = np.array([0, 0, 1, 1])

        # ========= decision variables ========= #
        self.x_ub = 5  # np.inf
        self.x_lb = - 5  # -np.inf

        # ========= calculate the parameters ========= #
        self.l_g_x = 1  # given any delta, L-constant of g
        self.l_f = 1  #
        self.l_g_delta = max(self.x_ub, 1 / (np.cos(1) ** 2))  # ## given any x, L-constant of g(x, delta)
        self.d_x = (self.x_ub - self.x_lb) * np.sqrt(self.x_dim)  # diameter of the domain x
        self.d_delta = 1  # diameter of delta
        self.r_delta = 0.5  #
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
        random_samples = np.random.uniform(0, 1, num_samples)

        arr_g_grad = - np.array([random_samples**i for i in range(self.dim_num_s)])  # Sum s^(i-1)

        arr_g_values = np.dot(x, arr_g_grad) - (- np.tan(random_samples))

        max_id = np.where(arr_g_values == arr_g_values.max())
        return arr_g_grad[:, max_id[0][0]], arr_g_values[max_id[0][0]]

    def adaptive_constraint_sampling(self, x, k=0, iteration=800):
        selected_samples = self.adaptive_selected_samples  # M; default is 1
        epsilon_k = (self.l_f + self.l_g_x) * self.d_x / np.sqrt(k + 1)
        # epsilon_k = 1 / np.sqrt(k + 1)
        kappa = np.minimum(np.minimum(epsilon_k / (2 * self.parameter_c), (epsilon_k / (2 * self.delta_dim))**2), 1)
        theta_sample = np.zeros([iteration])
        theta_rand = np.random.uniform(0, 1, iteration)
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
        for i in range(selected_samples):
            arr_g_grad, temp_values[i + 1] = self.calculate_g_grad_and_values(x, theta_rand[iteration - i - 1])
            if temp_values[i + 1].max() > temp0:
                arr_g_grad_max = arr_g_grad
                temp0 = temp_values[i + 1]
        return arr_g_grad_max, temp0

    def calculate_g_grad_and_values(self, x, theta_rand_value):
        g_grad = - np.array([theta_rand_value**i for i in range(self.dim_num_s)])
        return g_grad, np.dot(x, g_grad) - (- np.tan(theta_rand_value))

    def csa_algorithm(self):
        num_iteration = self.num_iterations
        # initial value of x_sol
        self.x_sol[:, 0] = self.x0
        if self.x_sol[:, 0].max() > self.x_ub or self.x_sol[:, 0].min() < self.x_lb:
            print('The initial value of x is invalid!')

        gamma = np.zeros(num_iteration + 1)
        eta = np.zeros(num_iteration + 1)

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
        'epsilon': 0.001,
        'num_iterations': 1000,
        'c_gamma': 0.05,
        'c_eta': 0.0010,
        'x0': np.array([2, 0, 0])
    }
    csa = CSA(parse_input)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    csa.run_adaptive_sampling(20, 30)
    # csa.run_fixed_sampling(1000)
    csa.plot_x_last_iterate()
    csa.plot_x_bar()
    plt.show()


if __name__ == '__main__':
    main()