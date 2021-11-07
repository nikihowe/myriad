import gin
import jax.numpy as jnp

from typing import Union, Optional

from myriad.systems import IndirectFHCS


@gin.configurable
class Harvest(IndirectFHCS):
  """
  Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 11, Lab 6)
  The model was was adapted from Wayne M. Getz. Optimal control and principles in population management.
  Proceedings of Symposia in Applied Mathematics, 30:63–82, 1984.

  This environment models the population level (scaled) of a population
  (for example, of vegetables) to be harvested.
  The time scale is too small for reproduction to occur, but the mass
  of each member of the population will grow over time following
  \\(\\frac{kt}{t+1}\\). The state ( \\(x\\) ) is the population level,
  while the control ( \\(u\\) ) is the harvest rate.
  We are trying to maximize:

  .. math::

    \\begin{align}
    & \\max_u \\quad && \\int_0^T A \\frac{kt}{t+1}x(t)u(t) - u^2(t) dt \\\\
    & \\; \\mathrm{s.t.}\\quad && x'(t) = -(m+u(t)) x(t) \\\\
    & && x(0)=x_0, \\; 0\\leq u(t) \\leq M, \\; A > 0
    \\end{align}
  """
  def __init__(self, A=5., k=10., m=.2, M=1., x_0=.4, T=10.):
    super().__init__(
      x_0=jnp.array([x_0]),   # Starting state
      x_T=None,               # Terminal state, if any
      T=T,                    # Duration of experiment
      bounds=jnp.array([      # Bounds over the states (x_0, x_1, ...) are given first,
        [jnp.NINF, jnp.inf],  # followed by bounds over controls (u_0,u_1, ...)
        [0, M],
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.adj_T = None  # Final condition over the adjoint, if any
    self.A = A
    """Nonnegative weight parameter"""
    self.k = k
    """Maximum mass of the species"""
    self.m = m
    """Natural death rate of the species"""
    self.M = M
    """Upper bound on harvesting that may represent physical limitations"""

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    d_x = -(self.m+u_t)*x_t

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return -1*self.A*(self.k*t/(t+1))*x_t*u_t + u_t**2  # Maximization problem converted to minimization

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return adj_t*(self.m+u_t) - self.A*(self.k*t/(t+1))*u_t

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char = 0.5*x_t * (self.A*(self.k*t/(t+1)) - adj_t)
    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
