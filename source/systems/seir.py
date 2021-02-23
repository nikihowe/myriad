import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from .base import FiniteHorizonControlSystem


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
      x_0 = jnp.array([self.S_0, self.E_0, self.I_0, self.N_0]),
      x_T = None,
      T = 20,
      bounds = jnp.array([
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [0.0, 1.0],
      ]),
      terminal_cost = False,
    )

  def dynamics(self, y_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
    S, E, I, N = y_t

    Ṡ = jnp.squeeze(self.b*N - self.d*S - self.c*S*I - u_t*S)
    Ė = jnp.squeeze(self.c*S*I - (self.e+self.d)*E)
    İ = jnp.squeeze(self.e*E - (self.g+self.a+self.d)*I)
    Ṅ = jnp.squeeze((self.b-self.d)*N - self.a*I)

    ẏ_t = jnp.array([Ṡ, Ė, İ, Ṅ])
    return ẏ_t
  
  def cost(self, y_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return self.A * y_t[2] + u_t ** 2

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   sns.set()
  #   plt.figure(figsize=(12, 2.5))
  #   ts_x = jnp.linspace(0, self.T, x.shape[0])
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   plt.subplot(151)
  #   plt.title('applied control')
  #   plt.plot(ts_u, u)
  #   plt.ylim(-0.1, 1.01)
  #
  #   for idx, title in enumerate(['S', 'E', 'I', 'N']):
  #     plt.subplot(1, 5, idx+2)
  #     plt.title(title)
  #     plt.plot(ts_x, x[:, idx])
  #
  #   plt.tight_layout()
  #   plt.show()
