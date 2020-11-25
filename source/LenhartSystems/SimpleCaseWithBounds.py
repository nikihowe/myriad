from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class SimpleCaseWithBounds(FiniteHorizonControlSystem):
    def __init__(self, A, C, M_1, M_2, x_0, T):
        """
                Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 9, Lab 4)
                A simple introductory environment example of the form :

                .. math::

                    \max_u \quad &\int_0^1 Ax(t) - u^2(t) dt \\
                    \mathrm{s.t.}\qquad & x'(t) = -\frac{1}{2}x^2(t) + Cu(t) \\
                    & x(0)=x_0>-2, \; A \geq 0, \; M_1 \leq u(t) \leq M_2

                :param A: Weight parameter
                :param C: Weight parameter
                :param M_1: Lower bound for the control
                :param M_2: Upper bound for the control
                :param x_0: Initial state
                :param T: Horizon
                """
        self.A = A
        self.C = C

        self.adj_T = None  # Final condition over the adjoint, if any

        super().__init__(
            x_0=np.array([x_0]),  # Starting state
            x_T=None,  # Terminal state, if any
            T=T,  # Duration of experiment
            bounds=np.array([  # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],  # followed by bounds over controls (u_0,u_1,...)
                [M_1, M_2],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray = None, t: np.ndarray = None) -> np.ndarray:
        d_x= -0.5*x_t**2+ self.C*u_t

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float:
        return -self.A*x_t + u_t**2 # Maximization problem converted to minimization

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return -self.A + x_t*adj_t

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = (self.C*adj_t)/2
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array = None, multi: bool = False) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

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

        if flag:
            plt.subplot(3, 1, 3)
            for adj_i in adj:
                plt.plot(ts_adj, adj_i)
            plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
            plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()