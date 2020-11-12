from ..systems import FiniteHorizonControlSystem

import jax.numpy as np
import gin
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class SimpleCase(FiniteHorizonControlSystem):
    def __init__(self, A, B, C, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 5, Lab 1)
        A simple introductory environment example of the form :

        .. math::

            \max_u \quad &\int_0^1 Ax(t) - Bu^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = -\frac{1}{2}x^2(t) + Cu(t) \\
            & x(0)=x_0>-2, \; A \geq 0, \; B > 0

        :param A: Weight parameter
        :param B: Weight parameter
        :param C: Weight parameter
        :param x_0: Initial state
        :param T: Horizon
        """
        self.A = A
        self.B = B
        self.C = C

        self.adj_T = None   # Final condition over the adjoint, if any

        super().__init__(
            x_0=np.array([x_0]),    # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [np.NINF, np.inf],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x= -0.5*x_t**2 + self.C*u_t

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*x_t - self.B*u_t**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return -self.A + x_t*adj_t

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = (self.C*adj_t)/(2*self.B)
        return np.minimum(self.bounds[0,1],np.maximum(self.bounds[0,0],char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))
        ts_x = np.linspace(0, self.T, x.shape[0])
        ts_u = np.linspace(0, self.T, u.shape[0])
        ts_adj = np.linspace(0, self.T, adj.shape[0])

        plt.subplot(3, 1, 1)
        plt.plot(ts_x, x)
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        plt.plot(ts_u, u)
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        plt.plot(ts_adj, adj)
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()