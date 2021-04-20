import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base import FiniteHorizonControlSystem


class CartPole(FiniteHorizonControlSystem):
  def __init__(self):
    """
    Cart-pole swing-up,
      from https://epubs.siam.org/doi/10.1137/16M1062569
    """

    # Physical parameters for the cart-pole example (Table 3)
    self.m1 = 1.0  # kg mass of cart
    self.m2 = 0.3  # kg mass of pole
    self.l = 0.5   # m pole length
    self.g = 9.81  # m/s^2 gravity acceleration
    self.u_max = 20  # N maximum actuator force
    self.d_max = 2.0 # m extent of the rail that cart travels on
    self.d = 1.0   # m distance traveled during swing-up

    super().__init__(
      x_0=jnp.array([0., 0., 0., 0.]),  # Starting state (Eq. 6.9)
      x_T=jnp.array([self.d, jnp.pi, 0., 0.]),  # Ending state (Eq. 6.9)
      T=2.0,  # s duration of swing-up,
      bounds=jnp.array([
        [-self.d_max, self.d_max],  # Eq. 6.7
        [-2*jnp.pi, 2*jnp.pi],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-self.u_max, self.u_max],  # Control bounds (Eq. 6.8)
      ]),
      terminal_cost=False,
    )

  # Cart-Pole Example: System Dynamics (Section 6.1)
  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    q1, q2, q̇1, q̇2 = x_t
    # Eq. 6.1
    q̈1 = ((self.l * self.m2 * jnp.sin(q2) * q̇2**2 + u_t + self.m2 * self.g * jnp.cos(q2) * jnp.sin(q2))
           / (self.m1 + self.m2 * (1 - jnp.cos(q2)**2)))
    q̈1 = jnp.squeeze(q̈1)
    # Eq. 6.2
    q̈2 = - ((self.l * self.m2 * jnp.cos(q2) * q̇2**2 + u_t * jnp.cos(q2) + (self.m1 + self.m2) * self.g * jnp.sin(q2))
             / (self.l * self.m1 + self.l * self.m2 * (1 - jnp.cos(q2)**2)))
    q̈2 = jnp.squeeze(q̈2)
    return jnp.array([q̇1, q̇2, q̈1, q̈2])
  
  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    # Eq. 6.3
    return u_t ** 2
  
  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   x = pd.DataFrame(x, columns=['q1', 'q2', 'q̈1', 'q̈2'])
  #
  #   # Plot optimal trajectory (Figure 10)
  #   sns.set(style='darkgrid')
  #   plt.figure(figsize=(9, 6))
  #   ts_x = jnp.linspace(0, self.T, x.shape[0])
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   plt.subplot(3, 1, 1)
  #   plt.ylabel('position (m)')
  #   plt.xlim(0, 2.01)
  #   plt.ylim(0, 1.5)
  #   plt.plot(ts_x, x['q1'], '-bo', clip_on=False, zorder=10)
  #
  #   plt.subplot(3, 1, 2)
  #   plt.ylabel('angle (rad)')
  #   plt.plot(ts_x, x['q2'], '-bo', clip_on=False, zorder=10)
  #   plt.xlim(0, 2.01)
  #   plt.ylim(-2, 4)
  #
  #   plt.subplot(3, 1, 3)
  #   plt.ylabel('force (N)')
  #   # plt.plot(ts_u, u, '-bo', clip_on=False, zorder=10)
  #   plt.step(ts_u, u, where="post", clip_on=False)
  #   plt.xlim(0, 2.01)
  #   plt.ylim(-20, 11)
  #
  #   plt.xlabel('time (s)')
  #   plt.tight_layout()
  #   plt.show()
