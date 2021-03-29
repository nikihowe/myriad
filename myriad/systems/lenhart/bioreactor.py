from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from myriad.systems import IndirectFHCS


@gin.configurable
class Bioreactor(IndirectFHCS):   # TODO: Add resolution for z state after optimization
  """
      Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 19, Lab 12)
      Additional information about this kind of model can be found in A. Heinricher, S. Lenhart, and A. Solomon.
      The application of optimal control methodology to a well-stirred bioreactor. Natural Resource Modeling, 9:61–80,
      1995.

      This environment is an example of a model where the cost is linear with respect to the control.
      It can still be solved by the FBSM algorithm since the optimal control are of the "bang-bang" type,
      i.e. it jumps from one boundary value to the other.

      This environment models the evolution of a bacteria population ( \\(x(t)\\) ) that helps in the degradation of a
      contaminant ( \\(z(t)\\) ) in the presence of a chemical nutrient ( \\(u(t)\\) ) that is added to boost the bacteria population
      growth. In this particular problem, the fact that only a terminal cost is associated to the state variable \\(z(t)\\)
      allows for the simplification of the problem into:

      .. math::

        \\begin{align}
        &\\max_{u} \\quad &&\\int_0^T Kx(t) - u(t) dt \\\\
        & \\; \\mathrm{s.t.}\\quad  &&x'(t) = Gu(t)x(t) - Dx^2(t) ,\\; x(0) = x_0 \\\\
        & && 0 \\leq u(t) \\leq M
        \\end{align}
      """
  def __init__(self, K=2., G=1., D=1., M=1., x_0=(.5, .1), T=2.):
    super().__init__(
      x_0=jnp.array([
        x_0[0],
      ]),                    # Starting state
      x_T=None,              # Terminal state, if any
      T=T,                   # Duration of experiment
      bounds=jnp.array([     # Bounds over the states (x_0, x_1 ...) are given first,
        [jnp.NINF, jnp.inf], # followed by bounds over controls (u_0, u_1,...)
        [0, M],
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.adj_T = None  # Final condition over the adjoint, if any
    self.K = K
    """Weight parameter"""
    self.G = G
    """Maximum growth rate of the bacteria population"""
    self.D = D
    """Natural  death rate of the bacteria population"""
    self.M = M
    """Physical limitation into the application of the chemical nutrient"""

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if u_t.ndim > 0:
      u_t, = u_t
    d_x = jnp.array([
      self.G*u_t*x_t[0] - self.D*x_t[0]**2
      ])

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return -self.K*x_t[0] + u_t  # Maximization problem converted to minimization

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return jnp.array([
      -self.K - self.G*u_t[0]*adj_t[0] + 2*self.D*x_t[0]*adj_t[0]
    ])

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    # bang-bang scenario
    temp = -1 + self.G*adj_t[:, 0]*x_t[:, 0]
    char = jnp.sign(temp.reshape(-1, 1)) * 2 * jnp.max(jnp.abs(self.bounds[-1])) + jnp.max(jnp.abs(self.bounds[-1]))

    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))
