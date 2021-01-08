from ..systems import IndirectFHCS
from ..config import SystemType
from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
class Cancer(IndirectFHCS):
    def __init__(self, r, a, delta, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 10, Lab 5)
        The model was originally described in K. Renee Fister and John Carl Panetta. Optimal control applied to
        competing chemotherapeutic cell-kill strategies. SIAM Journal of Applied Mathematics, 63(6):1954–71, 2003.

        The tumour is assumed to Gompertzian growth and the model follows a Skipper's log-kill hypothesis, that is, the
        cell-kill due to the chemotherapy treatment is proportional to the tumor population.

        This environment models the normalized density of a cancerous tumour undergoing chemotherapy. The state (x) is the
        normalized density of the tumour, while the control (u) is the strength of the drug used for chemotherapy.
        We are trying to minimize:

        .. math::

            \min_u \quad &\int_0^T ax^2(t) + u^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = rx(t)\ln \big( \frac{1}{x(t)} \big) - u(t)\delta x(t) \\
            & x(0)=x_0, \; u(t) \geq 0

        :param r: Growth rate of the tumor
        :param a: Positive weight parameter
        :param delta: Magnitude of the dose administered
        :param x_0: Initial normalized density of the tumor
        :param T: Horizon
        """
        super().__init__(
            _type=SystemType.CANCER,
            x_0=jnp.array([x_0]),   # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([      # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0, u_1,...)
                [0, jnp.inf],
            ]),
            terminal_cost=False,
            discrete=False,
        )

        self.adj_T = None  # Final condition over the adjoint, if any
        self.r = r
        self.a = a
        self.delta = delta

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        d_x = self.r * x_t * jnp.log(1 / x_t) - u_t * self.delta * x_t

        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return self.a*x_t**2 + u_t**2

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        return adj_t * (self.r + self.delta * u_t - self.r * jnp.log(1 / x_t)) - 2 * self.a * x_t

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char = 0.5*adj_t*self.delta*x_t
        return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray,
                      adj: Optional[jnp.ndarray] = None, 
                      other_x: Optional[jnp.ndarray] = None,
                      other_u: Optional[jnp.ndarray] = None,
                      save_title: Optional[str] = None) -> None:
        print("size of x", x.shape)
        print("size of u", u.shape)
        print("plotting solution")
        sns.set(style='darkgrid')
        plt.figure(figsize=(8, 9))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

        plan_with_node = False
        if other_u is not None and other_x is not None:
            plan_with_node = True

        adj = [adj]

        ts_x = jnp.linspace(0, self.T, x.shape[0])
        ts_u = jnp.linspace(0, self.T, u.shape[0])
        ts_adj = jnp.linspace(0, self.T, adj[0].shape[0])

        ax = plt.subplot(3, 1, 1)
        plt.plot(ts_x, x, "o-", label="True trajectory, using true optimal controls")
        if plan_with_node:
            plt.plot(ts_x, other_x, '.-', color="green", label="True trajectory, using controls calculated with NODE")
        elif other_x is not None:
            plt.plot(ts_x, other_x, '.-', color="green", label="NODE-Simulated trajectory, using optimal controls")
        ax.legend()
        plt.title("State of dynamic system")
        plt.ylabel("state (x)")

        ax = plt.subplot(3, 1, 2)
        plt.plot(ts_u, u, '-o', label="Planning with true dynamics")
        if other_u is not None:
            plt.plot(ts_u, other_u, '-o', label="Planning with NODE-Simulated dynamics")
        ax.legend()
        plt.title("Control of dynamic system")
        plt.ylabel("control (u)")

        if flag:
            plt.subplot(3, 1, 3)
            for adj_i in adj:
                plt.plot(ts_adj, adj_i)
            plt.title("Adjoint of dynamic system")
            plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()

        if save_title:
            plt.savefig(save_title)
        else:
            plt.show()