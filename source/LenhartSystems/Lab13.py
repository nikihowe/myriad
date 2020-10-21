from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab13Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    d_1: float    #
    d_2: float
    A: float    #
    guess_a: float    #
    guess_b: float


class Lab13(Lab13Parameters):
    def __init__(self, d_1=0.1, d_2=0.1, A=1, B=5, guess_a=-0.52, guess_b=-0.5, M=1, x_0=(10, 1, 0), T=10):
        self.adj_T = np.array([1, 0, 0]) # final condition over the adjoint

        super().__init__(
            d_1 =d_1,
            d_2 = d_2,
            A = A,
            guess_a = guess_a,
            guess_b = guess_b,
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
        )

    def update(self, caller):
        if caller.A: self.d_1 = caller.A
        if caller.B: self.d_2 = caller.B
        if caller.C: self.A = caller.C
        if caller.D: self.guess_a = caller.D
        if caller.E: self.guess_b = caller.E
        if caller.F:
            self.bounds =np.array([  # no bounds here
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, caller.F]
            ])
        if caller.G:
            self.x_T = [None, None, caller.G]
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0[0],
                caller.x_0[1],
                caller.x_0[2],
            ])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            (1 - x_t[1])*x_t[0] - self.d_1*x_t[0]*u_t[0],
            (x_t[0] - 1)*x_t[1] - self.d_2*x_t[1]*u_t[0],
            u_t[0],
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return A*0.5*u_t[0]**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            adj_t[0]*(x_t[1] -1 + self.d_1*u_t[0]) - adj_t[1]*x_t[1],
            adj_t[0]*x_t[0] + adj_t[1]*(1 - x_t[0] + self.d_2*u_t[0]),
            0
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char_0 = (adj_t[:, 0]*self.d_1*x_t[:, 0] + adj_t[:, 1]*self.d_2*x_t[:, 1] - adj_t[:, 2])/self.A
        char_0 = char_0.reshape(-1,1)

        return char_0

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