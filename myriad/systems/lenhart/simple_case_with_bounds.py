from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from myriad.systems import IndirectFHCS


@gin.configurable
class SimpleCaseWithBounds(IndirectFHCS):
  """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 9, Lab 4). \n
    A simple introductory environment example of the form:

    .. math::

      \\begin{align}
      & \\max_u \\quad && \\int_0^1 Ax(t) - u^2(t) dt \\\\
      & \\; \\mathrm{s.t.}\\quad && x'(t) = -\\frac{1}{2}x^2(t) + Cu(t) \\\\
      & && x(0)=x_0>-2, \\; A \\geq 0, \\; M_1 \\leq u(t) \\leq M_2
      \\end{align}
  """
  def __init__(self, A=1., C=4., M_1=-1., M_2=2., x_0=1., T=1.):
    super().__init__(
      x_0=jnp.array([x_0]),   # Starting state
      x_T=None,               # Terminal state, if any
      T=T,                    # Duration of experiment
      bounds=jnp.array([      # Bounds over the states (x_0, x_1 ...) are given first,
        # [jnp.NINF, jnp.inf],  # followed by bounds over controls (u_0,u_1,...)
        [0., 3.],  # changed based on observation of the true optimal trajectory using default M_1 and M_2
        [M_1, M_2],
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.A = A
    """Weight parameter"""
    self.C = C
    """Weight parameter"""
    self.M_1 = M_1
    """Lower bound for the control"""
    self.M_2 = M_2
    """Upper bound for the control"""
    self.adj_T = None  # Final condition over the adjoint, if any

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    d_x = -0.5*x_t**2 + self.C*u_t

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return -self.A*x_t + u_t**2  # Maximization problem converted to minimization

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return -self.A + x_t*adj_t

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char = (self.C*adj_t)/2
    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
