from typing import Union, Optional

import gin
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from source.systems import IndirectFHCS


@gin.configurable
class EpidemicSEIRN(IndirectFHCS):  # TODO : Add R calculation at the end
    def __init__(self, b, d, c, e, g, a, x_0, A, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 13, Lab 7)
        A typical SEIRN (or SEIR) model is considered here in order to find an optimal schedule for a vaccination
        campaign. Additional information about this model and some of its variations can be found in H. R. Joshi,
        S. Lenhart, M. Y. Li, and L. Wang. Optimal control methods applied to disease models. AMS Volume on Mathematical
        Studies on Human Disease Dynamics Emerging Paradigms and Challenges, 410:187–207, 2006

        The model contains multiples state varaibles; S(t) (x_0) is the number of individuals susceptible of contracting
        the disease at time t, while I(t) (x_2) and R(t) (x_3) are respectively the number of infectious and recovered
        (and immune) individuals. E(t) (x_1) is the number of individuals who have been exposed to the disease and are
        now in a latent state: they may develop the disease later on and become infectious, or they may simply become
        immune. N(t) (x_4) is the total population, i.e., the sum of all other states.
        The control is the vaccination rate among the susceptible individuals.
        Finally, note that all individuals are considered to be born susceptible. We want to minimize:

        .. math::

            \min_u \quad &\int_0^T A x_0(t) + u^2(t) dt \\
            \mathrm{s.t.}\qquad & x_0'(t) = bx_4(t) - dx_0(t) - cx_0(t)x_2(t) - u(t)x_0(t),\; x_0(0)\geq 0 \\
            & x_1'(t) = cx_0(t)x_2(t) - (e+d)x_1(t),\; x_1(0)\geq 0 \\
            & x_2'(t) = ex_1(t) - (g+a+d)x_2(t),\; x_2(0)\geq 0 \\
            & x_3'(t) = gx_2(t) - dx_3(t) + u(t)x_0(t),\; x_3(0)\geq 0 \\
            & x_4'(t) = (b-d)x_4(t) - ax_2(t),\; x_4(0)\geq 0 \\
            & 0\leq u(t) \leq 0.9 \; A > 0

        :param b: The exponential birth rate of the population
        :param d: The exponential death rate of the population
        :param c: The incidence rate of contamination
        :param e: The rate at which exposed individuals become contagious (1/e is the mean latent period)
        :param g: The recovery rate among infectious individuals (1/g is the mean infectious period)
        :param a: The death rate due to the disease
        :param x_0: The initial state, given by (S(t_0), E(t_0), I(t_0), R(t_0)
        :param A: Weight parameter balancing between the reduction of the infectious population and the vaccination cost
        :param T: Horizon
        """
        super().__init__(
            x_0=jnp.array([
                x_0[0],
                x_0[1],
                x_0[2],
                jnp.sum(x_0),
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [jnp.NINF, jnp.inf],
                [0, 0.9],
            ]),
            terminal_cost=False,
            discrete=False,
        )

        self.adj_T = None  # Final condition over the adjoint, if any
        self.b = b
        self.d = d
        self.c = c
        self.e = e
        self.g = g
        self.a = a
        self.A = A

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x_0, x_1, x_2, x_3 = x_t
        if u_t.ndim > 0:
            u_t, = u_t
        d_x = jnp.array([
            self.b*x_3 - self.d*x_0 - self.c*x_0*x_2 - u_t*x_0,
            self.c*x_0*x_2 - (self.e+self.d)*x_1,
            self.e*x_1 - (self.g+self.a+self.d)*x_2,
            (self.b-self.d)*x_3 - self.a*x_2
        ])
        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return self.A*x_t[2] + u_t**2

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        return jnp.array([
            adj_t[0]*(self.d+self.c*x_t[2]+u_t[0]) - adj_t[1]*self.c*x_t[2],
            adj_t[1]*(self.e+self.d) - adj_t[2]*self.e,
            -self.A + adj_t[0]*self.c*x_t[0] - adj_t[1]*self.c*x_t[0] + adj_t[2]*(self.g+self.a+self.d)
            + adj_t[3]*self.a,
            -self.b*adj_t[0] + adj_t[3]*(self.d-self.d)
        ])

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char = adj_t[:, 0]*x_t[:, 0]/2
        char = char.reshape(-1, 1)
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

        labels = ["Susceptible population", "Exposed population", "Infectious population", "Total population"]

        to_print = [2]  # curves we want to print out

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
