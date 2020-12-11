from ..systems import IndirectFHCS
from ..config import SystemType
from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
class HIVTreatment(IndirectFHCS):
    def __init__(self, s, m_1, m_2, m_3, r, T_max, k, N, x_0, A, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 14, Lab 8)
        Model adapted from : S. Butler, D. Kirschner, and S. Lenhart. Optimal control of chemotherapy affecting the
        infectivity of HIV. Advances in Mathematical Population Dynamics - Molecules, Cells and Man, 6:557–69, 1997.

        This model describes the the evolution of uninfected and infected (respectively x_0 and x_1) CD4⁺T cells, in the
        presence of free virus particles (x_2). The control is the administration of a chemotherapy drug that affects
        the infectivity of the virus. The goal is to maximize the number of uninfected CD4⁺T cells.
        Note that u(t) = 0 represents maximum therapy, while u(t) = 1 is no therapy. We want to maximize:

         .. math::

            \max_u \quad &\int_0^T A x_0(t) - (1-u(t))^2 dt \\
            \mathrm{s.t.}\qquad & x_0'(t) = \frac{s}{1+x_2(t)} - m_1x_0(t) + rx_0(t)\big[1 - \frac{x_0(t)+x_1(t)}{T_{\mathrm{max}}} \big],\; x_0(0)> 0 \\
            & x_1'(t) = u(t)kx_2(t)x_0(t) - m_2x_1(t),\; x_1(0)> 0 \\
            & x_2'(t) = Nm_2x_1(t) - m_3x_2(t),\; x_2(0)> 0 \\
            & 0\leq u(t) \leq 1 \; A > 0

        :param s: Parameter varying the rate of generation of new CD4⁺T cells
        :param m_1: Natural death rate of uninfected CD4⁺T cells
        :param m_2: Natural death rate of infected CD4⁺T cells
        :param m_3: Natural death rate of free virus particles
        :param r: Growth rate of CD4⁺T cells per day
        :param T_max: Maximum growth of CD4⁺T cells
        :param k: Rate of infection among CD4⁺T cells from free virus particles
        :param N: Average number of virus particles produced before the CD4⁺T host cell dies.
        :param x_0: Initial state (x_0, x_1, x_2)
        :param A: Weight parameter balancing the cost
        :param T: Horizon
        """
        super().__init__(
            _type=SystemType.HIVTREATMENT,
            x_0=jnp.array([
                x_0[0],
                x_0[1],
                x_0[2],
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [0, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [0, jnp.inf],
                [0, jnp.inf],
                [0, 1],
            ]),
            terminal_cost=False,
            discrete=False,
        )

        self.adj_T = None  # Final condition over the adjoint, if any
        self.s = s
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3
        self.r = r
        self.T_max = T_max
        self.k = k
        self.N = N
        self.A = A

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x_0, x_1, x_2 = x_t
        if u_t.ndim > 0:
            u_t, = u_t
        d_x = jnp.array([
            self.s/(1+x_2) - self.m_1*x_0 + self.r*x_0*(1-(x_0+x_1)/self.T_max) - u_t*self.k*x_0*x_2,
            u_t*self.k*x_0*x_2 - self.m_2*x_1,
            self.N*self.m_2*x_1 - self.m_3*x_2,
            ])

        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return -self.A*x_t[0] + (1-u_t)**2  # Maximization problem converted to minimization

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        return jnp.array([
            -self.A + adj_t[0]*(self.m_1 - self.r*(1-(x_t[0]+x_t[1])/self.T_max) + self.r*x_t[0]/self.T_max
                                + u_t[0]*self.k*x_t[2]) - adj_t[1]*u_t[0]*self.k*x_t[2],
            adj_t[0]*self.r*x_t[0]/self.T_max + adj_t[1]*self.m_2 - adj_t[2]*self.N*self.m_2,
            adj_t[0]*(self.s/(1+x_t[2])**2 + u_t[0]*self.k*x_t[0]) - adj_t[1]*u_t[0]*self.k*x_t[0] + adj_t[2]*self.m_3,
        ])

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char = 1 + 0.5*self.k*x_t[:, 0]*x_t[:, 2]*(adj_t[:, 1]-adj_t[:, 0])
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

        labels = ["Healthy cells", "Infected cells", "Viral charge"]

        to_print = [0]  # curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            if idx in [0]:
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
