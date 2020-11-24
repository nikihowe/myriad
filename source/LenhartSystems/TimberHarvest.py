from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class TimberHarvest(FiniteHorizonControlSystem):
    def __init__(self, r, k, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 18, Lab 11)
        Additional information can be found in Morton I. Kamien and Nancy L. Schwartz. Dynamic Optimization:
        The Calculus of Variations and Optimal Control in Economics and Management. North-Holland, New York, 1991.

        This environment is an example of model where the cost is linear w/r to the control. It can still be solved by
        the FBSM algorithm since the optimal control are of the "bang-bang" type, i.e. it jumps from one boundary value
        to the other.

        In this problem we are trying to optimize the tree harvesting in a timber farm, resulting in the production of
        raw timber (x(t)). The harvest percentage over the land is low enough that we can assumed that there will always
        be enough mature trees ready for harvest. The timbers are sold right after their production, generating a income
        proportional to the production at every time t. The operators then have the choice of reinvesting a fraction of
        this revenue directly into the plant (u(t)), thus stimulating future production. But, this reinvestment come at the
        price of loosing potential interest return over the period T is the revenue had been placed. The control problem
        is therefore:

        .. math::

            \max_{u} \quad &\int_0^T e^{-rt}x(t)[1 - u(t)] dt \\
            \mathrm{s.t.}\qquad & x'(t) = kx(t)u(t) ,\; x(0) > 0 \\
            & 0 \leq u(t) \leq 1

        :param r: Discount rate encouraging investment early on
        :param k: Return constant of reinvesting into the plant, taking into account cost of labor and land
        :param x_0: Initial raw timber production
        :param T: Horizon
        """
        self.adj_T = None # Final condition over the adjoint, if any
        self.r = r
        self.k = k

        super().__init__(
            x_0=np.array([
                x_0,
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [0, 1],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray = None, t: np.ndarray = None) -> np.ndarray:
        d_x = np.asarray([
            self.k*x_t[0]*u_t[0]
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float:
        return -np.exp(-self.r*t)*x_t[0]*(1-u_t[0])  # Maximization problem converted to minimization

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            u_t[0]*(np.exp(-self.r*t[0])-self.k*adj_t[0]) - np.exp(-self.r*t[0])
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        #bang-bang scenario
        temp = x_t[:,0]*(self.k*adj_t[:,0] - np.exp(-self.r*t[:,0]))
        char = np.sign(temp.reshape(-1,1)) * 2*np.max(np.abs(self.bounds[-1])) + np.max(np.abs(self.bounds[-1]))  #arithmetic bang-bang

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        # debug : #TODO remove after making adj correctly an option
        if adj is None:
            adj = u.copy()  # Only for testing #TODO remove after test
            flag = False
        else:
            flag = True

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Timber harvested"]

        to_print = [0] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i, label='Reinvestment level')
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        if flag:
            plt.subplot(3, 1, 3)
            for idx, adj_i in enumerate(adj):
                if idx in to_print:
                    plt.plot(ts_adj, adj_i)
            plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
            plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()