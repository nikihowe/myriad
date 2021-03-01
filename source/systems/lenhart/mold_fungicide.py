from typing import Union, Optional

import gin
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from source.systems import IndirectFHCS


@gin.configurable
class MoldFungicide(IndirectFHCS):
  def __init__(self, r=0.3, M=10, A=10, x_0=1.0, T=5):
    """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 6, Lab 2)
    This environment models the concentration level of a mould population that we try to control by
    applying a fungicide. The state (x) is the population concentration, while the control (u) is
    the amount of fungicide added. We are trying to minimize:
    .. math::
      \min_u \quad &\int_0^T Ax^2(t) + u^2(t) dt \\
      \mathrm{s.t.}\qquad & x'(t) = r(M - x(t)) - u(t)x(t) \\
      & x(0)=x_0 \;
    :param r: Growth rate
    :param M: Carrying capacity
    :param A: Weight parameter, balancing between controlling the population and limiting the fungicide use
    :param x_0: Initial mold population concentration
    :param T: Horizon
    """
    super().__init__(
      x_0=jnp.array([x_0]),  # Starting state
      x_T=None,              # Terminal state, if any
      T=T,                   # Duration of experiment
      bounds=jnp.array([     # Bounds over the states (x_0, x_1 ...) are given first,
        [jnp.NINF, jnp.inf],    # followed by bounds over controls (u_0,u_1,...)
        [jnp.NINF, jnp.inf],
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.adj_T = None  # Final condition over the adjoint, if any
    self.r = r
    self.M = M
    self.A = A

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
         v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    d_x = self.r*(self.M - x_t) - u_t*x_t

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return self.A*x_t**2 + u_t**2

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
        t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return adj_t*(self.r + u_t) - 2*self.A*x_t

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                 t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char = 0.5*adj_t*x_t
    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
