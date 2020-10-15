from dataclasses import dataclass
from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Lab8Parameters(FiniteHorizonControlSystem): #TODO : describe variables
    s: float    #
    m_1: float    #
    m_2: float    #
    m_3: float    #
    r: float    #
    T_max: float    #
    k: float
    N: float
    A: float    # weight parameter of the objective

class Lab8(Lab8Parameters):  #TODO : Add R calculation at the end
    def __init__(self, s=10, m_1=0.02, m_2=0.5, m_3=4.4, r=0.03, T_max=1500, k=0.000024, N=300, x_0=(800,0.04,1.5), A=0.05, T=20):
        self.adj_T = None # final condition over the adjoint

        super().__init__(
            s = s,
            m_1 = m_1,
            m_2 = m_2,
            m_3 = m_3,
            r = r,
            T_max = T_max,
            k = k,
            N = N,
            A = A,
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
        )

    def update(self, caller):
        if caller.A: self.s = caller.A
        if caller.B: self.m_1 = caller.B
        if caller.C: self.m_2 = caller.C
        if caller.D: self.m_3 = caller.D
        if caller.E: self.r = caller.E
        if caller.F: self.T_max = caller.F
        if caller.G: self.k = caller.G
        if caller.H: self.N = caller.H
        if caller.I: self.A = caller.I
        if caller.x_0:
            self.x_0 = np.array([
                caller.x_0[0],
                caller.x_0[1],
                caller.x_0[2],
            ])
        if caller.T: self.T = caller.T

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