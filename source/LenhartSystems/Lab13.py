from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Lab13(FiniteHorizonControlSystem):
    def __init__(self, d_1, d_2, A, B, guess_a, guess_b, M, x_0, T):
        self.adj_T = np.array([1, 0, 0]) # final condition over the adjoint
        self.d_1 = d_1
        self.d_2 = d_2
        self.A = A
        self.guess_a = guess_a
        self.guess_b = guess_b

        super().__init__(
            x_0=np.array([
                x_0[0],
                x_0[1],
                x_0[2]
            ]),  # Starting state
            x_T=[None, None, B],
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, M]
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            (1 - x_t[1])*x_t[0] - self.d_1*x_t[0]*u_t[0],
            (x_t[0] - 1)*x_t[1] - self.d_2*x_t[1]*u_t[0],
            u_t[0],
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*0.5*u_t[0]**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            adj_t[0]*(x_t[1] -1 + self.d_1*u_t[0]) - adj_t[1]*x_t[1],
            adj_t[0]*x_t[0] + adj_t[1]*(1 - x_t[0] + self.d_2*u_t[0]),
            0
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = (adj_t[:, 0]*self.d_1*x_t[:, 0] + adj_t[:, 1]*self.d_2*x_t[:, 1] - adj_t[:, 2])/self.A
        char = char.reshape(-1,1)

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Prey Population", "Predator Population"]

        to_print = [0,1] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i, label='Pesticide level')
        plt.legend()
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