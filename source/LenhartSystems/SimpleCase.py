from ..systems import IndirectFHCS
from ..config import SystemType
from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
class SimpleCase(IndirectFHCS):
    def __init__(self, A, B, C, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 5, Lab 1)
        A simple introductory environment example of the form :

        .. math::

            \max_u \quad &\int_0^1 Ax(t) - Bu^2(t) dt \\
            \mathrm{s.t.}\qquad & x'(t) = -\frac{1}{2}x^2(t) + Cu(t) \\
            & x(0)=x_0>-2, \; A \geq 0, \; B > 0

        :param A: Weight parameter
        :param B: Weight parameter
        :param C: Weight parameter
        :param x_0: Initial state
        :param T: Horizon
        """
        super().__init__(
            _type=SystemType.SIMPLECASE,
            x_0=jnp.array([x_0]),    # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=jnp.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [jnp.NINF, jnp.inf],      # followed by bounds over controls (u_0,u_1,...)
                [jnp.NINF, jnp.inf],
            ]),
            terminal_cost=False,
            discrete=False,
        )

        self.A = A
        self.B = B
        self.C = C
        self.adj_T = None  # Final condition over the adjoint, if any

    def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                 v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        d_x = -0.5*x_t**2 + self.C*u_t

        return d_x

    def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
        return -self.A*x_t + self.B*u_t**2  # Maximization problem converted to minimization

    def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
                t: Optional[jnp.ndarray]) -> jnp.ndarray:
        return -self.A + x_t*adj_t

    def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                               t: Optional[jnp.ndarray]) -> jnp.ndarray:
        char = (self.C*adj_t)/(2*self.B)
        return jnp.minimum(self.bounds[0, 1], jnp.maximum(self.bounds[0, 0], char))

    def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 12))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

        ts_x = jnp.linspace(0, self.T, x.shape[0])
        ts_u = jnp.linspace(0, self.T, u.shape[0])
        ts_adj = jnp.linspace(0, self.T, adj.shape[0])

        plt.subplot(3, 1, 1)
        plt.plot(ts_x, x)
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        plt.plot(ts_u, u)
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        if flag:
            plt.subplot(3, 1, 3)
            plt.plot(ts_adj, adj)
            plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
            plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()
