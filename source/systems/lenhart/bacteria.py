from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from source.systems import IndirectFHCS


@gin.configurable
class Bacteria(IndirectFHCS):
  def __init__(self, r=1., A=1., B=12., C=1., x_0=1.):
    """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 7, Lab 3)

    This environment models the concentration level of a bacteria population that we try to control by providing
    a chemical nutrient that stimulates growth. However, the use of the chemical leads to the production of
    a chemical byproduct by the bacteria that in turn hinders growth. The state (x) is the bacteria population
    concentration, while the control (u) is the amount of chemical nutrient added. We are trying to maximize:

    .. math::

      \max_u \quad &Cx(1) - \int_0^1 u^2(t) dt \\
      \mathrm{s.t.}\qquad & x'(t) = rx(t) + Au(t)x(t) - Bu^2(t)e^{-x(t)} \\
      & x(0)=x_0, \; A,B,C \geq 0

    :param r: Growth rate
    :param A: Relative strength of the chemical nutrient
    :param B: Strength of the byproduct
    :param C: Payoff associated to the final bacteria population concentration
    :param x_0: Initial bacteria population concentration
    """
    super().__init__(
      x_0=jnp.array([x_0]),   # Starting state
      x_T=None,               # Terminal state, if any
      T=1,                    # Duration of experiment
      bounds=jnp.array([      # Bounds over the states (x_0, x_1 ...) are given first,
        [jnp.NINF, jnp.inf],  # followed by bounds over controls (u_0, u_1,...)
        [jnp.NINF, jnp.inf],
      ]),
      terminal_cost=True,
      discrete=False,
    )

    self.adj_T = jnp.array([C])  # Final condition over the adjoint, if any
    self.r = r
    self.A = A
    self.B = B
    self.C = C

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    d_x = self.r * x_t + self.A * u_t * x_t - self.B * u_t ** 2 * jnp.exp(-x_t)

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return u_t**2   # Maximization problem converted to minimization

  def terminal_cost_fn(self, x_T: Optional[jnp.ndarray], u_T: Optional[jnp.ndarray],
                       T: Optional[jnp.ndarray] = None) -> float:
    return -self.C*x_T.squeeze() # squeeze is necessary for using SHOOTING

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return -adj_t*(self.r + self.A * u_t + self.B * u_t ** 2 * jnp.exp(-x_t))

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char = adj_t*self.A*x_t/(2 * (1 + self.B * adj_t * jnp.exp(-x_t)))
    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
