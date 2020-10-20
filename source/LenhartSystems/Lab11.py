from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab11Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    r: float    #
    k: float

class Lab11(Lab11Parameters):
    def __init__(self, r=0, k=1, x_0=100, T=5):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            r = r,
            k = k,
            x_0=np.array([
                x_0,
            ]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, 1],
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.r = caller.A
        if caller.B: self.k = caller.B
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0
            ])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            self.k*x_t[0]*u_t[0]
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return np.exp(-self.r*t[0])*x_t[0]*(1-u_t[0])

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            u_t[0]*(np.exp(-self.r*t[0])-self.k*adj_t[0]) - np.exp(-self.r*t[0])
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        #bang-bang scenario
        temp = x_t[:,0]*(self.k*adj_t[:,0] - np.exp(-self.r*t[:,0]))
        char = np.sign(temp.reshape(-1,1)) * 2*np.max(np.abs(self.bounds[-1])) + np.max(np.abs(self.bounds[-1]))  #arithmetic bang-bang

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Timber harvested"]

        to_print = [0] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i, label='Reinvestment level')
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