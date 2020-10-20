from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab10Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    a: float    #
    b: float
    c: float    #
    A: float    #
    l: float    #


class Lab10(Lab10Parameters):
    def __init__(self, a=1, b=1, c=1, A=2, l=0.5, x_0=(0.75,0), T=0.2):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            a = a,
            b = b,
            c = c,
            A = A,
            l = l,
            x_0=np.array([
                x_0[0],
                x_0[1],
            ]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.a = caller.A
        if caller.B: self.b = caller.B
        if caller.C: self.c = caller.C
        if caller.D: self.A = caller.D
        if caller.E: self.l = caller.E
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0[0],
                caller.x_0[1],
            ])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            -self.a*x_t[0] -self.b*x_t[1],
            -self.c*x_t[1] + u_t[0]
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return A*(x_t[0]-l)**2 + u_t[0]**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            -2*self.A*(x_t[0]-self.l) +adj_t[0]*self.a,
            adj_t[0]*self.b + adj_t[1]*self.c
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char_0 = -adj_t[:,1]/2
        char_0 = char_0.reshape(-1,1)

        return char_0

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Blood Glucose", "Net Hormonal Concentration"]

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
            plt.plot(ts_u, u_i, label='Insulin level')
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