from ..systems import IndirectFHCS
from ..config import SystemType
from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
class BearPopulations(IndirectFHCS):
    def __init__(self, r, K, m_p, m_f, c_p, c_f, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 15, Lab 9)
        Additional reference can be found in R. A. Salinas, S. Lenhart, and L. J. Gross. Control of a metapopulation
        harvesting model for black bears. Natural Resource Modeling, 18:307–21, 2005.

        The model represents the metapopulation of black bears, i.e. a population consisting of multiple local
        populations, which can interact with each other. In this particular scenario, the author models the
        bear population density in a park (protected) area (x_0), a forest area (x_1) and a urban area (x_2). Natural
        reproduction happens only inside the park and forest area, and the goal is to limit the bear population that
        migrates to the urban area.
        The control is a harvesting rate (hunting) that occurs inside the forest area and, with bigger cost, in the
        park area. The goal is thus to minimize:

        .. math::

            \min_{u_p,u_f} \quad &\int_0^T x_2(t) + c_p u_p(t)^2 + c_f u_f(t)^2  dt \\
            \mathrm{s.t.}\qquad & x_0'(t) = rx_0(t) - \frac{r}{K}x_0(t)^2 + \frac{m_f r}{K}\big( 1 - \frac{x_0(t)}{K} \big)x_1(t)^2 - u_p(t)x_0(t),\; x_0(0)\geq 0 \\
            & x_1'(t) = rx_1(t) - \frac{r}{K}x_1(t)^2 + \frac{m_p r}{K}\big( 1 - \frac{x_1(t)}{K} \big)x_0(t)^2 - u_f(t)x_1(t),\; x_1(0)\geq 0 \\
            & x_2'(t) = r(1-m_p)\frac{x_0(t)^2}{K} + r(1-m_f)\frac{x_1(t)^2}{K} + \frac{m_f r}{K^2}x_0(t)x_1(t)^2 + \frac{m_p r}{K^2}x_0(t)^2x_1(t)^,\; x_2(0)\geq 0 \\
            & 0\leq u_p(t) \leq 1 \; 0\leq u_f(t) \leq 1

        :param r: Population growth rate
        :param K: Carrying capacity of the areas (density wise)
        :param m_p: Proportion of the park boundary connected to the forest areas
        :param m_f: Proportion of the forest areas connected to the park area
        :param c_p: Cost associated with harvesting in the park
        :param c_f: Cost associated with harvesting in the forest
        :param x_0: Initial state (x_0, x_1, x_2)
        :param T: Horizon
        """
        super().__init__(
            _type=SystemType.BEARPOPULATIONS,
            x_0=jnp.array([
                x_0[0],
                x_0[1],
                x_0[2],
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [0, 1],
                [0, 1],
            ]),
            terminal_cost=False,
            discrete=False,
        )

        self.adj_T = None  # Final condition over the adjoint, if any
        self.r = r
        self.K = K
        self.m_p = m_p
        self.m_f = m_f
        self.c_p = c_p
        self.c_f = c_f

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        k = self.r/self.K
        k2 = self.r/self.K**2
        x_0, x_1, x_2 = x_t
        u_0, u_1 = u_t

        d_x = jnp.array([
            self.r*x_0 - k*x_0**2 + k*self.m_f*(1-x_0/self.K)*x_1**2 - u_0*x_0,
            self.r*x_1 - k*x_1**2 + k*self.m_p*(1-x_1/self.K)*x_0**2 - u_1*x_1,
            k*(1-self.m_p)*x_0**2 + k*(1-self.m_f)*x_1**2 + k2*self.m_f*x_0*x_1**2 + k2*self.m_p*(x_0**2)*x_1,
            ])

        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return x_t[2] + self.c_p*u_t[0]**2 + self.c_f*u_t[1]**2

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        k = self.r / self.K
        k2 = self.r / self.K**2

        return jnp.array([
            adj_t[0]*(2*k*x_t[0] + k2*self.m_f*x_t[1]**2 + u_t[0] - self.r)
            - adj_t[1]*(2*k*self.m_p*(1-x_t[1]/self.K)*x_t[0])
            + adj_t[2]*(2*k*(self.m_p-1)*x_t[0] - k2*self.m_f*x_t[1]**2 - 2*k2*self.m_p*x_t[0]*x_t[1]),
            adj_t[1]*(2*k*x_t[1] + k2*self.m_p*x_t[0]**2 + u_t[1] - self.r)
            - adj_t[0]*(2*k*self.m_f*(1-x_t[0]/self.K)*x_t[1])
            + adj_t[2]*(2*k*(self.m_f-1)*x_t[1] - 2*k2*self.m_f*x_t[0]*x_t[1] - k2*self.m_p*x_t[0]**2),
            -1,
        ])

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char_0 = adj_t[:, 0]*x_t[:, 0]/(2*self.c_p)
        char_0 = char_0.reshape(-1, 1)
        char_0 = jnp.minimum(self.bounds[-2, 1], jnp.maximum(self.bounds[-2, 0], char_0))

        char_1 = adj_t[:, 1] * x_t[:, 1]/(2 * self.c_f)
        char_1 = char_1.reshape(-1, 1)
        char_1 = jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char_1))

        return jnp.hstack((char_0, char_1))

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

        labels = ["Park bear pop", "Forest bear pop", "Outside bear pop"]

        to_print = [0, 1, 2]    # curves we want to print out

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
