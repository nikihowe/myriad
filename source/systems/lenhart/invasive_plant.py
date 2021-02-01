from typing import Union, Optional

import gin
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from source.systems import IndirectFHCS


@gin.configurable
class InvasivePlant(IndirectFHCS):
    def __init__(self, B, k, eps, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 24, Lab 14)
        This problem was first look at in M. E. Moody and R. N. Mack. Controlling the spread of plant invasions:
        the importance of nascent foci. Journal of Applied Ecology, 25:1009–21, 1988.
        The general formulation that the we look at in this environment was presented in A. J. Whittle, S. Lenhart, and
        L. J. Gross. Optimal control for management of an invasive plant species. Mathematical Biosciences and
        Engineering, to appear, 2007.

        The scenario considered in this environment has been modified from its original formulation so
        so that the state terminal cost term is linear instead of quadratic. Obviously, the optimal solutions are
        different from the original problem, but the model behavior is similar.

        In this environment, we look at the growth of an invasive species that has a main focus population (x_i) and
        4 smaller satellite populations (x_{i\neq j}. The area occupied by the different population are assumed to be
        circular, with a growth that can be represented via the total radius of the population area. Annual interventions
        are made after the growth period, removing a ratio of the population radius (u_{j,t}). Since the interventions are
        annual, we are in a discrete time model. We aim to:

        .. math::

            \min_{u} \quad &\sum_{j=0}^4 \bigg[x_{j,T} + B\sum_{t=0}^{T-1} u_{j,t}^2 \bigg] \\
            \mathrm{s.t.}\qquad & x_{j,t+1} = \bigg( x_{j,t} + \frac{k x_{j,t}}{\epsilon + x_{j,t}}\bigg) (1-u_{j,t}) ,\; x_{j,0} = \rho_j \\
            & 0 \leq u_{j,t} \leq 1

        :param B: Positive weight parameter
        :param k: Spread rate of the population
        :param eps: Small constant, used to scale the spread by :math:`\frac{r}{\epsilon+r}` so eradication is possible
        :param x_0: Initial radius of the different populations
        :param T: Horizon
        """
        super().__init__(
            x_0=jnp.array(x_0),      # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ]),
            terminal_cost=False,
            discrete=True,
        )

        self.adj_T = jnp.ones(5)  # Final condition over the adjoint, if any
        self.B = B
        self.k = k
        self.eps = eps

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        next_x = (x_t + x_t*self.k/(self.eps + x_t)) * (1 - u_t)

        return next_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return self.B*(u_t**2).sum()

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        prev_adj = adj_t * (1-u_t) * (1 + self.eps*self.k/(self.eps + x_t)**2)

        return prev_adj

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        shifted_adj = adj_t[1:, :]
        shifted_x_t = x_t[:-1, :]
        char = 0.5*shifted_adj/self.B * (shifted_x_t + shifted_x_t*self.k/(self.eps + shifted_x_t))

        return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))   # same bounds for all control

    def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = jnp.linspace(0, self.T, x[0].shape[0])
        ts_u = jnp.linspace(0, self.T - 1, u[0].shape[0])
        ts_adj = jnp.linspace(0, self.T, adj[0].shape[0])

        labels = ["Focus 1", "Focus 2", "Focus 3", "Focus 4", "Focus 5"]

        to_print = [0, 1, 2, 3, 4]  # curves we want to print out

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
