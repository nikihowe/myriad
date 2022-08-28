import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from myriad.custom_types import Params
from myriad.systems import FiniteHorizonControlSystem


class VanDerPol(FiniteHorizonControlSystem):
  def __init__(self, a=1.):
    """
    Driven Van der Pol oscillator,
      from http://casadi.sourceforge.net/v1.8.0/users_guide/html/node8.html
    """
    self.a = a

    super().__init__(
      x_0=jnp.array([0., 1.]),
      x_T=jnp.zeros(2),
      T=10.0,
      bounds=jnp.array([
        # [-jnp.inf, jnp.inf],  # state 1
        # [-jnp.inf, jnp.inf],  # state 2
        [-4., 4.],  # state 1 (from observation)
        [-4., 4.],  # state 2 (from observation)
        [-0.75, 1.0],  # control
      ]),
      terminal_cost=False,
    )

  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    x0, x1 = x_t
    _x0 = jnp.squeeze(self.a * (1. - x1 ** 2) * x0 - x1 + u_t)
    _x1 = jnp.squeeze(x0)
    return jnp.asarray([_x0, _x1])

  def parametrized_dynamics(self, params: Params, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    a = params['a']
    x0, x1 = x_t
    _x0 = jnp.squeeze(a * (1. - x1 ** 2) * x0 - x1 + u_t)
    _x1 = jnp.squeeze(x0)
    return jnp.asarray([_x0, _x1])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return x_t.T @ x_t + u_t ** 2

  def parametrized_cost(self, params: Params, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return x_t.T @ x_t + u_t ** 2  # nothing to learn here!

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   x = pd.DataFrame(x, columns=['x0', 'x1'])
  #
  #   sns.set(style='darkgrid')
  #   plt.figure(figsize=(9, 4))
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   plt.subplot(1, 2, 1)
  #   plt.plot(x['x0'], x['x1'])
  #
  #   plt.subplot(1, 2, 2)
  #   plt.step(ts_u, u, where="post")
  #   plt.xlabel('time (s)')
  #
  #   plt.tight_layout()
  #   plt.show()
