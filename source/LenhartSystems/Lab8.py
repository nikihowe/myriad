from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Lab8(FiniteHorizonControlSystem):
    def __init__(self, s, m_1, m_2, m_3, r, T_max, k, N, x_0, A, T):
        self.adj_T = None # final condition over the adjoint
        self.s = s
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.r = r
        self.T_max = T_max
        self.k = k
        self.N = N
        self.A = A

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
                [0, 1],  # Control bounds
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            self.s/(1+x_t[2]) - self.m_1*x_t[0] + self.r*x_t[0]*(1-(x_t[0]+x_t[1])/self.T_max) - u_t[0]*self.k*x_t[0]*x_t[2],
            u_t[0]*self.k*x_t[0]*x_t[2] - self.m_2*x_t[1],
            self.N*self.m_2*x_t[1] - self.m_3*x_t[2],
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*x_t[0] - (1-u_t)**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            -self.A + adj_t[0]*(self.m_1 - self.r*(1-(x_t[0]+x_t[1])/self.T_max) + self.r*x_t[0]/self.T_max + u_t[0]*self.k*x_t[2]) - adj_t[1]*u_t[0]*self.k*x_t[2],
            adj_t[0]*self.r*x_t[0]/self.T_max + adj_t[1]*self.m_2 - adj_t[2]*self.N*self.m_2,
            adj_t[0]*(self.s/(1+x_t[2])**2 + u_t[0]*self.k*x_t[0]) - adj_t[1]*u_t[0]*self.k*x_t[0] + adj_t[2]*self.m_3,
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = 1 + 0.5*self.k*x_t[:,0]*x_t[:,2]*(adj_t[:,1]-adj_t[:,0])
        char = char.reshape(-1,1)
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Healthy cells", "Infected cells", "Viral charge"]

        to_print = [2] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            if idx in [0]:
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