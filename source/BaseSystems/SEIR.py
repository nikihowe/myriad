from ..systems import FiniteHorizonControlSystem
from ..config import SystemType
import jax.numpy as jnp
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt


class SEIR(FiniteHorizonControlSystem):
  def __init__(self):
    self.b = 0.525
    self.d = 0.5
    self.c = 0.0001
    self.e = 0.5

    self.g = 0.1
    self.a = 0.2

    self.S_0 = 1000.0
    self.E_0 = 100.0
    self.I_0 = 50.0
    self.R_0 = 15.0
    self.N_0 = self.S_0 + self.E_0 + self.I_0 + self.R_0

    self.A = 0.1
    self.M = 1000

    super().__init__(
      _type=SystemType.SEIR,
      x_0=jnp.array([self.S_0, self.E_0, self.I_0, self.N_0]),
      x_T=None,
      T=20,
      bounds=jnp.array([
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [0.0, 1.0],
      ]),
      terminal_cost=False,
    )

  def dynamics(self, y_t: jnp.ndarray, u_t: jnp.float64, t: jnp.ndarray = None) -> jnp.ndarray:
    s, e, i, n = y_t

    s_dot = jnp.squeeze(self.b * n - self.d * s - self.c * s * i - u_t * s)
    e_dot = jnp.squeeze(self.c * s * i - (self.e + self.d) * e)
    i_dot = jnp.squeeze(self.e * e - (self.g + self.a + self.d) * i)
    n_dot = jnp.squeeze((self.b - self.d) * n - self.a * i)

    y_dot_t = jnp.array([s_dot, e_dot, i_dot, n_dot])
    return y_dot_t

  def cost(self, y_t: jnp.ndarray, u_t: jnp.float64, t: float = None) -> jnp.float64:
    return self.A * y_t[2] + u_t ** 2

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, other_x: Optional[jnp.ndarray] = None) -> None:
    sns.set()
    plt.figure(figsize=(12, 2.5))
    ts_x = jnp.linspace(0, self.T, x.shape[0])
    ts_u = jnp.linspace(0, self.T, u.shape[0])

    plt.subplot(151)
    plt.title('applied control')
    plt.plot(ts_u, u)
    plt.ylim(-0.1, 1.01)

    for idx, title in enumerate(['S', 'E', 'I', 'N']):
      plt.subplot(1, 5, idx + 2)
      plt.title(title)
      plt.plot(ts_x, x[:, idx])
      if other_x is not None:
        plt.plot(ts_u, other_x[:, idx])

    plt.tight_layout()
    plt.show()
