from ..systems import IndirectFHCS
from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
class FishHarvest(IndirectFHCS):
    def __init__(self, A, k, m, M, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 11, Lab 6)
        The model was was adapted from Wayne M. Getz. Optimal control and principles in population management.
        Proceedings of Symposia in Applied Mathematics, 30:63–82, 1984.

        This environment model the population level (scaled) of a fish population to be harvested, and that was
        introduced into a fishery of some kind (so the initial level of population is known). The time before harvesting
        is too small for reproduction to occur, but the average mass of the fish will grow over time following
        :math:`\frac{kt}{t+1}`. The state (x) is the population level, while the control (u) is the harvest rate.
        We are trying to maximize:

        .. math::

            \max_u \quad &\int_0^T A \frac{kt}{t+1}x(t)u(t) - u^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = -(m+u(t)) x(t) \\
            & x(0)=x_0, \; 0\leq u(t) \leq M \; A > 0

        :param A: Nonnegative weight parameter
        :param k: Maximum mass of the fish species
        :param m: Natural death rate of the fish
        :param M: Upper bound on harvesting that may represent physical limitations
        :param x_0: Initial fish population level
        :param T: Horizon
        """
        self.adj_T = None   # Final condition over the adjoint, if any
        self.A = A
        self.k = k
        self.m = m
        self.M = M

        super().__init__(
            x_0=jnp.array([x_0]),    # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [0, M],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        d_x = -(self.m+u_t)*x_t

        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return -1*self.A*(self.k*t/(t+1))*x_t*u_t + u_t**2  # Maximization problem converted to minimization

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        return adj_t*(self.m+u_t) - self.A*(self.k*t/(t+1))*u_t

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char = 0.5*x_t * (self.A*(self.k*t/(t+1)) - adj_t)
        return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 12))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

        x, u, adj = x.T, u.T, adj.T

        ts_x = jnp.linspace(0, self.T, x[0].shape[0])
        ts_u = jnp.linspace(0, self.T, u[0].shape[0])
        ts_adj = jnp.linspace(0, self.T, adj[0].shape[0])

        plt.subplot(3, 1, 1)
        for x_i in x:
            plt.plot(ts_x, x_i*(self.k*ts_x/(ts_x+1)))
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
