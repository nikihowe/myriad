from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab6Parameters(FiniteHorizonControlSystem):
    A: float    # weight parameter of the objective
    k: float    # maximum mass of the fish species
    m: float    # natural death rate
    M: float    # upper bound for harvesting; must be 0 <= M <= 1

class Lab6(Lab6Parameters):
    def __init__(self, A=5.0, k=10.0, m=0.2, M=1.0, x_0=0.4, T=10 ):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            A = A,
            k = k,
            m = m,
            M = M,
            x_0=np.array([x_0]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, M],  # Control bounds
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.A = caller.A  # Growth rate of the tumor
        if caller.B: self.k = caller.B          # Positive weight parameter
        if caller.C: self.m = caller.C   # Magnitude of the chemo dose
        if caller.x_0: self.x_0 = np.array([caller.x_0])
        if caller.T: self.T = caller.T
        if caller.M: self.bounds = np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, caller.M],  # Control bounds
            ])

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x= -(self.m+u_t)*x_t

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*(self.k*t/(t+1))*x_t*u_t - u_t**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return adj_t*(self.m+u_t) - self.A *(self.k*t/(t+1))*u_t

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = 0.5*x_t * (self.A*(self.k*t/(t+1)) - adj_t)
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        plt.subplot(3, 1, 1)
        for x_i in x:
            plt.plot(ts_x, x_i *(self.k*ts_x/(ts_x+1)))
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for u_i in u:
            plt.plot(ts_u, u_i)
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        for adj_i in adj:
            plt.plot(ts_adj, adj_i)
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()