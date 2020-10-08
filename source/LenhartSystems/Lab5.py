from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab5Parameters(FiniteHorizonControlSystem):
    r: float
    a: float
    delta: float

class Lab5(Lab5Parameters):
    def __init__(self, r=0.3, a=3, delta=0.45, x_0=0.975, T=20 ):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            r = r,
            a = a,
            delta = delta,
            x_0=np.array([x_0]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [0, np.inf],  # Control bounds
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.r = caller.A  # Growth rate of the tumor
        if caller.B: self.a = caller.B          # Positive weight parameter
        if caller.C: self.delta = caller.C   # Magnitude of the chemo dose
        if caller.x_0: self.x_0 = np.array([caller.x_0])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        d_x= self.r*x_t*np.log(1/x_t) - u_t*self.delta*x_t

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.a*x_t**2 + u_t**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        return adj_t*(self.r + self.delta*u_t - self.r*np.log(1/x_t)) - 2*self.a*x_t

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        char = 0.5*adj_t*self.delta*x_t
        return np.minimum(self.bounds[0, 1], np.maximum(self.bounds[0, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array, multi: bool = False) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        if not multi:
            x, u, adj = [x], [u], [adj]

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        plt.subplot(3, 1, 1)
        for x_i in x:
            plt.plot(ts_x, x_i)
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