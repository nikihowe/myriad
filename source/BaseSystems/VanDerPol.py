from ..systems import FiniteHorizonControlSystem
from ..config import SystemType
import jax.numpy as jnp


class VanDerPol(FiniteHorizonControlSystem):
  def __init__(self):
    super().__init__(
      _type=SystemType.VANDERPOL,
      x_0=jnp.array([0., 1.]),
      x_T=jnp.zeros(2),
      T=10.0,
      bounds=jnp.array([
        [-jnp.inf, jnp.inf],  # state 1
        [-jnp.inf, jnp.inf],  # state 2
        [-0.75, 1.0],  # control
      ]),
      terminal_cost=False,
    )

  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: jnp.ndarray = None) -> jnp.ndarray:
    x0, x1 = x_t
    _x0 = jnp.squeeze((1. - x1 ** 2) * x0 - x1 + u_t)
    _x1 = jnp.squeeze(x0)
    return jnp.asarray([_x0, _x1])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return x_t.T @ x_t + u_t ** 2
