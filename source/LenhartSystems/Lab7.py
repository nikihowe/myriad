from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab7Parameters(FiniteHorizonControlSystem):
    b: float    # natural exponential birth rate
    d: float    # natural exponential death rate
    c: float    # rate of contamination
    e: float    # rate at which exposed individuals become infectious
    g: float    # rate of recovery among infectious people
    a: float    # death rate due to the infection
    A: float    # weight parameter of the objective

class Lab7(Lab7Parameters):  #TODO : Add R calculation at the end
    def __init__(self, b=0.525, d=0.5, c=0.0001, e=0.5, g=0.1, a=0.2, x_0=(1000,100,50,15), A=0.1, T=20 ):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            b = b,
            d = d,
            c = c,
            e = e,
            g = g,
            a = a,
            A = A,
            x_0=np.array([
                x_0[0],
                x_0[1],
                x_0[2],
                np.sum(x_0),
            ]),  # Starting state
            x_T=None,
            T=T,  # duration of experiment
            bounds=np.array([  # no bounds here
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, 0.9],  # Control bounds
            ]),
            terminal_cost=False,
        )

    def update(self, caller):
        if caller.A: self.a = caller.A
        if caller.B: self.b = caller.B
        if caller.C: self.c = caller.C
        if caller.D: self.d = caller.D
        if caller.E: self.e = caller.E
        if caller.F: self.A = caller.F
        if caller.G: self.g = caller.G
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0[0],
                caller.x_0[1],
                caller.x_0[2],
                np.sum(caller.x_0),
            ])
        if caller.T: self.T = caller.T

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x= np.asarray([
            self.b*x_t[3] - self.d*x_t[0] - self.c*x_t[0]*x_t[2] - u_t[0]*x_t[0],
            self.c*x_t[0]*x_t[2] - (self.e+self.d)*x_t[1],
            self.e*x_t[1] - (self.g+self.a+self.d)*x_t[2],
            (self.b-self.d)*x_t[3] - self.a*x_t[2]
        ])
        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*x_t[2] + u_t**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            adj_t[0]*(self.d+self.c*x_t[2]+u_t[0]) - adj_t[1]*self.c*x_t[2],
            adj_t[1]*(self.e+self.d) - adj_t[2]*self.e,
            -self.A + adj_t[0]*self.c*x_t[0] - adj_t[1]*self.c*x_t[0] + adj_t[2]*(self.g+self.a+self.d) + adj_t[3]*self.a,
            -self.b*adj_t[0] + adj_t[3]*(self.d-self.d)
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = adj_t[:,0]*x_t[:,0]/2
        char = char.reshape(-1, 1)
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Susceptible population", "Exposed population", "Infectious population", "Total population"]

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