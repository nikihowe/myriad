from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Bioreactor(FiniteHorizonControlSystem): #TODO: Add resolution for z state after optimization
    def __init__(self, K, G, D, M, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 19, Lab 12)
        Additional information about this kind of model can be found in A. Heinricher, S. Lenhart, and A. Solomon.
        The application of optimal control methodology to a well-stirred bioreactor. Natural Resource Modeling, 9:61–80,
        1995.

        This environment is an example of model where the cost is linear w/r to the control. It can still be solved by
        the FBSM algorithm since the optimal control are of the "bang-bang" type, i.e. it jumps from one boundary value
        to the other.

        This environment try to model the evolution of a bacteria population (x(t)) that helps in the degradation of a
        contaminant (z(t)) in the presence of a chemical nutrient (u(t)) that is added to boost the bacteria population
        growth. In this particular problem, the fact that only a terminal cost is associated to the state variable z(t)
        allows for the simplification of the problem into:

        .. math::

            \max_{u} \quad &\int_0^T Kx(t) - u(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = Gu(t)x(t) - Dx^2(t) ,\; x(0) = x_0 \\
            & 0 \leq u(t) \leq M

        :param K: Weight parameter
        :param G: Maximum growth rate of the bacteria population
        :param D: Natural  death rate of the bacteria population
        :param M: Physical limitation into the application of the chemical nutrient
        :param x_0: Initial bacteria concentration
        :param T: Horizon
        """
        self.adj_T = None # Final condition over the adjoint, if any
        self.K = K
        self.G = G
        self.D = D

        super().__init__(
            x_0=np.array([
                x_0[0],
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [0, M],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            self.G*u_t[0]*x_t[0] - self.D*x_t[0]**2
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.K*x_t[0] - u_t[0]

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            -self.K - self.G*u_t[0]*adj_t[0] + 2*self.D*x_t[0]*adj_t[0]
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        #bang-bang scenario
        temp = -1 + self.G*adj_t[:,0]*x_t[:,0]
        char = np.sign(temp.reshape(-1,1)) * 2*np.max(np.abs(self.bounds[-1])) + np.max(np.abs(self.bounds[-1]))  #arithmetic bang-bang

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Bacteria Concentration"]

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
            plt.plot(ts_u, u_i, label='Nutrient Injection')
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