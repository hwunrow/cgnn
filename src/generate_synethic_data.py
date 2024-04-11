import numpy as np
import matplotlib.pyplot as plt
import pickle
import inspect
import logging


class simulate_data:
    def __init__(
        self,
        rt_0,
        rt_1,
        midpoint,
        k,
        n_t,
        t_I,
        N,
        S0,
        I0,
        run_deterministic=False,
        **kwargs,
    ):
        r"""
        Args:
          rt_0 (float): Rt value before midpoint
          rt_1 (float): Rt value after midpoint
          midpoint (int): Sigmoid midpoint
          k (float): Logistic growth rate
          n_t (int): Number of days
          t_I (float):
          N (int): Population
          I0 (int): Intitial number of infectors
          run_deterministic (bool): Flag to simulate deterministically
        """

        if isinstance(midpoint, float):
            logging.info("Casting midpoint to int")
            midpoint = int(midpoint)
        if isinstance(n_t, float):
            logging.info("Casting n_t to int")
            n_t = int(n_t)
        if isinstance(N, float):
            logging.info("Casting N to int")
            N = int(N)
        if isinstance(I0, float):
            logging.info("Casting I0 to int")
            I0 = int(I0)

        self.true_params = {
            "rt_0": rt_0,
            "rt_1": rt_1,
            "midpoint": midpoint,
            "k": k,
            "n_t": n_t,
            "t_I": t_I,
            "N": N,
            "S0": S0,
            "I0": I0,
        }
        self.rt_0 = rt_0
        self.rt_1 = rt_1
        self.midpoint = midpoint
        self.k = k
        self.n_t = n_t
        self.run_deterministic = run_deterministic

        self.t_I = t_I
        self.N = N
        self.I0 = I0
        self.S0 = N - I0

        self.rt = self.sigmoid(rt_0, rt_1, midpoint, k, n_t)

        if run_deterministic:
            self.S, self.I, self.R, self.i = self.gen_sir_det(
                self.rt, t_I, N, S0, I0, n_t
            )
            self.i_true = self.i
        else:
            self.S, self.I, self.R, self.i, self.i_true = self.gen_sir_stoch(
                self.rt, t_I, N, S0, I0, n_t, **kwargs
            )

    def sigmoid(self, rt_0, rt_1, midpoint, k, n_t):
        """Computes sigmoid curve"""
        t = np.arange(0, n_t)
        sigmoid = rt_0 + (rt_1 - rt_0) / (1 + np.exp(-k * (t - midpoint)))
        return sigmoid

    def construct_beta(self, rt, t_I):
        self.beta = rt / t_I
        return rt / t_I

    def gen_sir_det(self, rt, t_I, N, S0, I0, n_t):
        beta = self.construct_beta(rt, t_I)
        S = np.array([S0])
        Ir = np.array([I0])
        R = np.array([0])
        i = np.array([0])
        for t in range(n_t):
            dSI = beta[t] * Ir[t] * S[t] / N
            dIR = Ir[t] / t_I

            S_new = np.clip(S[t] - dSI, 0, N)
            I_new = np.clip(Ir[t] + dSI - dIR, 0, N)
            R_new = np.clip(R[t] + dIR, 0, N)

            S = np.append(S, S_new)
            Ir = np.append(Ir, I_new)
            R = np.append(R, R_new)
            i = np.append(i, dSI)

        return S, Ir, R, i

    def gen_sir_stoch(
        self, rt, t_I, N, S0, I0, n_t, add_noise=False, noise_param=1 / 25
    ):
        beta = self.construct_beta(rt, t_I)
        S = np.array([S0])
        Ir = np.array([I0])
        R = np.array([0])
        i = np.array([0])
        for t in range(n_t):
            dSI = np.random.poisson(beta[t] * Ir[t] * S[t] / N)
            dIR = np.random.poisson(Ir[t] / t_I)

            S_new = np.clip(S[t] - dSI, 0, N)
            I_new = np.clip(Ir[t] + dSI - dIR, 0, N)
            R_new = np.clip(R[t] + dIR, 0, N)

            S = np.append(S, S_new)
            Ir = np.append(Ir, I_new)
            R = np.append(R, R_new)
            i = np.append(i, dSI)

        i_true = i
        if add_noise:
            i = i.astype("float64")
            self.noise_param = noise_param
            obs_error_var = np.maximum(1.0, i[1:] ** 2 * noise_param)
            obs_error_sample = np.random.normal(0, 1, size=n_t)
            i[1:] += obs_error_sample * np.sqrt(obs_error_var)
            i = np.clip(i, 0, N)

        return S, Ir, R, i, i_true

    def compute_data_distribution(self, num_real=300):
        data_distribution = np.zeros(shape=self.i.shape)

        for _ in range(num_real):
            a = simulate_data(**self.true_params, add_noise=True, noise_param=1 / 50)
            data_distribution = np.vstack((data_distribution, a.i))

        data_distribution = np.delete(data_distribution, (0), axis=0)
        self.data_distribution = data_distribution.T

    def plot_rt(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.rt, ".-")
        ax.set_title("Rt")

    def plot_SIR(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.S, ".-", label="S")
        ax.plot(self.I, ".-", label="I")
        ax.plot(self.R, ".-", label="R")
        ax.set_title("Stochastic SIR")
        ax.legend()

    def plot_obs(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.i, ".")
        ax.set_title("Stochastic Daily Case Counts")

    def plot_moving_average(self, ax=None):
        def _moving_average(a, n=7):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1 :] / n

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(_moving_average(np.diff(self.i, n=1), n=7))
        ax.hlines(0, 0, 365, color="red")
        ax.vlines(100, -160, 100, color="red")

    def plot_all(self, path=None, name="synthetic_plots"):
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
        self.plot_rt(axs[0, 0])
        self.plot_SIR(axs[0, 1])
        self.plot_obs(axs[1, 0])
        fig.delaxes(axs[1, 1])

        if path:
            plt.savefig(f"{path}/{name}.pdf")

    def save_data(self, path=None, name="synthetic_data"):
        # log source code
        lines = inspect.getsource(simulate_data)
        logging.info(lines)
        with open(f"{path}/{name}.pkl", "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
