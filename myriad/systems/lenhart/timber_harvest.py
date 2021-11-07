from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from myriad.custom_types import Params
from myriad.systems import IndirectFHCS


@gin.configurable
class TimberHarvest(IndirectFHCS):
  """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 18, Lab 11)
    Additional information can be found in Morton I. Kamien and Nancy L. Schwartz. Dynamic Optimization:
    The Calculus of Variations and Optimal Control in Economics and Management. North-Holland, New York, 1991.

    This environment is an example of model where the cost is linear with respect to the control.
    It can still be solved by the FBSM algorithm since the optimal control are of the "bang-bang" type,
    i.e., it jumps from one boundary value to the other.

    In this problem we are trying to optimize tree harvesting in a timber farm, resulting in the production of
    raw timber ( \\(x(t)\\) ). The harvest percentage over the land
    is low enough that we can assume that there will always
    be sufficiently many mature trees ready for harvest. The timber is sold immediately after production,
    generating a income proportional to the production at every time t. The operators then have the choice of
    reinvesting a fraction of this revenue directly into the plant ( \\(u(t)\\) ), thus stimulating future production.
    But, this reinvestment comes at the price of losing potential interest over the period T if the
    revenue were saved. The control problem is therefore:

    .. math::

      \\begin{align}
      & \\max_{u} \\quad && \\int_0^T e^{-rt}x(t)[1 - u(t)] dt \\\\
      & \\mathrm{s.t.}\\quad && x'(t) = kx(t)u(t) ,\\; x(0) > 0 \\\\
      & && 0 \\leq u(t) \\leq 1
      \\end{align}
  """

  def __init__(self, r=0., k=1., x_0=100., T=5.):
    super().__init__(
      x_0=jnp.array([
        x_0,
      ]),  # Starting state
      x_T=None,  # Terminal state, if any
      T=T,  # Duration of experiment
      bounds=jnp.array([  # Bounds over the states (x_0, x_1 ...) are given first,
        [0., 20_000],  # followed by bounds over controls (u_0,u_1,...)
        [0., 1.],  # nh added the bounds
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.adj_T = None  # Final condition over the adjoint, if any
    self.r = r
    """Discount rate encouraging investment early on"""
    self.k = k
    """Return constant of reinvesting into the plant, taking into account cost of labor and land"""

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if u_t.ndim > 0:
      u_t, = u_t
    d_x = jnp.array([
      self.k * x_t[0] * u_t
    ])

    return d_x

  def parametrized_dynamics(self, params: Params, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                            v_t: Optional[Union[float, jnp.ndarray]] = None,
                            t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    k = params['k']
    if u_t.ndim > 0:
      u_t, = u_t
    d_x = jnp.array([
      k * x_t[0] * u_t
    ])

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return -jnp.exp(-self.r * t) * x_t[0] * (1 - u_t)  # Maximization problem converted to minimization

  def parametrized_cost(self, params: Params, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                        t: Optional[jnp.ndarray] = None) -> float:
    return -jnp.exp(-self.r * t) * x_t[0] * (1 - u_t)  # not learning cost function for now

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return jnp.array([
      u_t[0] * (jnp.exp(-self.r * t[0]) - self.k * adj_t[0]) - jnp.exp(-self.r * t[0])
    ])

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    # bang-bang scenario
    temp = x_t[:, 0] * (self.k * adj_t[:, 0] - jnp.exp(-self.r * t[:, 0]))
    char = jnp.sign(temp.reshape(-1, 1)) * 2 * jnp.max(jnp.abs(self.bounds[-1])) + jnp.max(jnp.abs(self.bounds[-1]))

    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
