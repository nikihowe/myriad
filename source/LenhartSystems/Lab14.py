from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

from jax import vmap
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab14Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    B: float    #
    k: float
    eps: float    #

class Lab14(Lab14Parameters):
    def __init__(self, B=1, k=1, eps=0.01, x_0=(0.5, 1, 1.5, 2, 10), T=10):
        self.adj_T = np.ones(5) # final condition over the adjoint

        super().__init__(
            B=B,
            k=k,
            eps=eps,
            x_0=np.array(x_0),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ]),
            terminal_cost=False,
            discrete=True,
        )

    def update(self, caller):
        if caller.A: self.B = caller.A
        if caller.B: self.k = caller.B
        if caller.C: self.eps = caller.C
        if caller.x_0:
            self.x_0 = np.array(caller.x_0)
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        next_x = (x_t + x_t*self.k/(self.eps + x_t)) * (1 - u_t)

        return next_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.B*(u_t**2).sum()

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        prev_adj = adj_t * (1-u_t) * (1 + self.eps*self.k/(self.eps + x_t)**2)

        return  prev_adj

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        shifted_adj = adj_t[1:,:]
        shifted_x_t = x_t[:-1, :]
        char = 0.5*shifted_adj/self.B * (shifted_x_t + shifted_x_t*self.k/(self.eps + shifted_x_t))

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char)) # bounds are the same for all control

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T-1, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Focus 1", "Focus 2", "Focus 3", "Focus 4", "Focus 5"]

        to_print = [0,1,2,3,4] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, 'o', label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i, 'o', label='Focus ratio cropped')
        plt.legend()
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        for idx, adj_i in enumerate(adj):
            if idx in to_print:
                plt.plot(ts_adj, adj_i, "o")
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()