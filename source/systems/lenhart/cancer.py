from typing import Union, Optional

import gin
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from source.systems import IndirectFHCS


@gin.configurable
class Cancer(IndirectFHCS):
    def __init__(self, r=0.3, a=3, delta=0.45, x_0=0.975, T=20):
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
            x_0=jnp.array([x_0]),   # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([      # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
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

    def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 12))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

        x, u, adj = [x], [u], [adj]

        ts_x = jnp.linspace(0, self.T, x[0].shape[0])
        ts_u = jnp.linspace(0, self.T, u[0].shape[0])
        ts_adj = jnp.linspace(0, self.T, adj[0].shape[0])

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
