from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Lab9(FiniteHorizonControlSystem):
    def __init__(self, r, K, m_p, m_f, c_p, c_f, x_0, T):
        self.adj_T = None # final condition over the adjoint
        self.r = r
        self.K = K
        self.m_p = m_p
        self.m_f = m_f
        self.c_p = c_p
        self.c_f = c_f

        super().__init__(
            x_0=np.array([
                x_0[0],
                x_0[1],
                x_0[2],
            ]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, 1],
                [0, 1],# Control bounds
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        k = self.r/self.K
        k2 = self.r/self.K**2

        d_x = np.asarray([
            self.r*x_t[0] - k*x_t[0]**2 + k*self.m_f*(1-x_t[0]/self.K)*x_t[1]**2 - u_t[0]*x_t[0],
            self.r*x_t[1] - k*x_t[1]**2 + k*self.m_p*(1-x_t[1]/self.K)*x_t[0]**2 - u_t[1]*x_t[1],
            k*(1-self.m_p)*x_t[0]**2 + k*(1-self.m_f)*x_t[1]**2 + k2*self.m_f*x_t[0]*x_t[1]**2 + k2*self.m_p*(x_t[0]**2)*x_t[1],
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return x_t[2] + self.c_p*u_t[0]**2 + self.c_f*u_t[1]**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        k = self.r / self.K
        k2 = self.r / self.K**2

        return np.array([
            adj_t[0]*(2*k*x_t[0] + k2*self.m_f*x_t[1]**2 + u_t[0] - self.r) - adj_t[1]*(2*k*self.m_p*(1-x_t[1]/self.K)*x_t[0]) + adj_t[2]*(2*k*(self.m_p-1)*x_t[0] - k2*self.m_f*x_t[1]**2 - 2*k2*self.m_p*x_t[0]*x_t[1]),
            adj_t[1]*(2*k*x_t[1] + k2*self.m_p*x_t[0]**2 + u_t[1] - self.r) - adj_t[0]*(2*k*self.m_f*(1-x_t[0]/self.K)*x_t[1]) + adj_t[2]*(2*k*(self.m_f-1)*x_t[1] - 2*k2*self.m_f*x_t[0]*x_t[1] - k2*self.m_p*x_t[0]**2),
            -1,
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char_0 = adj_t[:,0]*x_t[:,0]/(2*self.c_p)
        char_0 = char_0.reshape(-1,1)
        char_0 = np.minimum(self.bounds[-2, 1], np.maximum(self.bounds[-2, 0], char_0))

        char_1 = adj_t[:, 1] * x_t[:, 1]/(2 * self.c_f)
        char_1 = char_1.reshape(-1, 1)
        char_1 = np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char_1))


        return np.hstack((char_0,char_1))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Park bear pop", "Forest bear pop", "Outside bear pop"]

        to_print = [0,1,2] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i)
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        for idx, adj_i in enumerate(adj):
            if idx in to_print:
                plt.plot(ts_adj, adj_i)
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()