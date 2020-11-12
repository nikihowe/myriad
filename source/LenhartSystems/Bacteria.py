from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Bacteria(FiniteHorizonControlSystem):
    def __init__(self, r, A, B, C, x_0):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 7, Lab 3)

        This environment model the concentration level of a bacteria population that we try to control by providing
        a chemical nutrient that stimulate growth. However, the use of the chemical leads to the production of
        a chemical byproduct by the bacteria that in turn hinders growth. The state (x) is the bacteria population
        concentration, while the control (u) is the amount of chemical nutrient added. We are trying to maximize:

        .. math::
            \max_u \quad &Cx(1) - \int_0^1 u^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = rx(t) + Au(t)x(t) - Bu^2(t)e^{-x(t)} \\
            & x(0)=x_0, \; A,B,C \geq 0

        :param r: Growth rate
        :param A: Relative strength of the chemical nutrient
        :param B: Strength of the byproduct
        :param C: Payoff associated to the final bacteria population concentration
        :param x_0: Initial bacteria population concentration
        """
        self.adj_T = np.array([C])  # Final condition over the adjoint, if any
        self.r = r
        self.A = A
        self.B = B

        super().__init__(
            x_0=np.array([x_0]),    # Starting state
            x_T=None,               # Terminal state, if any
            T=1,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [np.NINF, np.inf],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x= self.r*x_t + self.A*u_t*x_t - self.B*u_t**2*np.exp(-x_t)

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return u_t**2 #TODO: Terminal reward missing

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return -adj_t*(self.r+self.A*u_t+self.B*u_t**2*np.exp(-x_t))

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = adj_t*self.A*x_t/(2*(1+self.B*adj_t*np.exp(-x_t)))
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

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
        plt.title("Optimal bacteria concentration of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for u_i in u:
            plt.plot(ts_u, u_i)
        plt.title("Optimal use of chemical nutrient in dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        for adj_i in adj:
            plt.plot(ts_adj, adj_i)
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()