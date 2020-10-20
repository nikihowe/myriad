from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab12Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    K: float    #
    G: float
    D: float

class Lab12(Lab12Parameters): #TODO: Add resolution for z state after optimization
    def __init__(self, K=2, G=1, D=1, M=1, x_0=(0.5,0.1), T=2):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            K = K,
            G = G,
            D = D,
            x_0=np.array([
                x_0[0],
            ]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, M],
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.K = caller.A
        if caller.B: self.G = caller.B
        if caller.C: self.D = caller.C
        if caller.D:
            self.bounds = np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, caller.D],
            ])
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0[0],
            ])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            self.G*u_t[0]*x_t[0] - self.D*x_t[0]**2
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.K*x_t[0] - u_t[0]

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            -self.K - self.G*u_t[0]*adj_t[0] + 2*self.D*x_t[0]*adj_t[0]
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        #bang-bang scenario
        temp = -1 + self.G*adj_t[:,0]*x_t[:,0]
        char = np.sign(temp.reshape(-1,1)) * 2*np.max(np.abs(self.bounds[-1])) + np.max(np.abs(self.bounds[-1]))  #arithmetic bang-bang

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Bacteria Concentration"]

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
            plt.plot(ts_u, u_i, label='Nutrient Injection')
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