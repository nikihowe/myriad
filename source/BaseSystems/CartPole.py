from ..systems import FiniteHorizonControlSystem
from ..config import SystemType
import jax.numpy as jnp


class CartPole(FiniteHorizonControlSystem):
  def __init__(self):
    # Physical parameters for the cart-pole example (Table 3)
    self.m1 = 1.0     # kg mass of cart
    self.m2 = 0.3     # kg mass of pole
    self.l = 0.5      # m pole length
    self.g = 9.81     # m/s^2 gravity acceleration
    self.u_max = 20   # N maximum actuator force
    self.d_max = 2.0  # m extent of the rail that cart travels on
    self.d = 1.0      # m distance traveled during swing-up

    super().__init__(
      _type=SystemType.CARTPOLE,
      x_0=jnp.array([0., 0., 0., 0.]),          # Starting state (Eq. 6.9)
      x_T=jnp.array([self.d, jnp.pi, 0., 0.]),  # Ending state (Eq. 6.9)
      T=2.0,                                    # s duration of swing-up,
      bounds=jnp.array([
        [-self.d_max, self.d_max],  # Eq. 6.7
        [-2 * jnp.pi, 2 * jnp.pi],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-self.u_max, self.u_max],  # Control bounds (Eq. 6.8)
      ]),
      terminal_cost=False,
    )

  # Cart-Pole Example: System Dynamics (Section 6.1)
  # Note that the dynamics are time-independent
  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: jnp.ndarray = None) -> jnp.ndarray:
    q1, q2, q_dot_1, q_dot_2 = x_t
    # Eq. 6.1
    q_dot_dot_1 = ((self.l * self.m2 * jnp.sin(q2) * q_dot_2 ** 2 + u_t + self.m2 * self.g * jnp.cos(q2) * jnp.sin(q2))
           / (self.m1 + self.m2 * (1 - jnp.cos(q2) ** 2))).squeeze()
    # Eq. 6.2
    q_dot_dot_2 = - (
            (self.l * self.m2 * jnp.cos(q2) * q_dot_2 ** 2 + u_t * jnp.cos(q2) + (self.m1 + self.m2) * self.g * jnp.sin(q2))
            / (self.l * self.m1 + self.l * self.m2 * (1 - jnp.cos(q2) ** 2))).squeeze()
    return jnp.array([q_dot_1, q_dot_2, q_dot_dot_1, q_dot_dot_2])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    # Eq. 6.3
    return u_t ** 2
