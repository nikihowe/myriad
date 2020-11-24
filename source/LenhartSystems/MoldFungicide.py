from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class MoldFungicide(FiniteHorizonControlSystem):
    def __init__(self, r, M, A, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 6, Lab 2)

        This environment model the concentration level of a mold population that we try to control by
        applying a fungicide. The state (x) is the population concentration, while the control (u) is
        the amount of fungicide added. We are trying to minimize:

        .. math::

            \min_u \quad &\int_0^T Ax^2(t) + u^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = r(M - x(t)) - u(t)x(t) \\
            & x(0)=x_0 \;

        :param r: Growth rate
        :param M: Carrying capacity
        :param A: Weight parameter, balancing between controlling the population and limiting the fungicide use
        :param x_0: Initial mold population concentration
        :param T: Horizon
        """
        self.adj_T = None   # Final condition over the adjoint, if any
        self.r = r
        self.M = M
        self.A = A

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

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray=None, t: np.ndarray=None) -> np.ndarray:
        d_x= self.r*(self.M - x_t) - u_t[0]*x_t

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray=None) -> float:
        return self.A*x_t**2 + u_t**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return adj_t*(self.r + u_t) - 2*self.A*x_t

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = 0.5*adj_t*x_t
        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array=None, multi: bool = False) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        # debug : #TODO remove after making adj correctly an option
        if adj is None:
            adj = u.copy()  # Only for testing #TODO remove after test
            flag=False
        else: flag=True

        if not multi:
            x, u, adj = [x], [u], [adj]

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        plt.subplot(3, 1, 1)
        for x_i in x:
            plt.plot(ts_x, x_i)
        plt.title("Optimal mold population of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for u_i in u:
            plt.plot(ts_u, u_i)
        plt.title("Optimal use of fungicide system via forward-backward sweep")
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