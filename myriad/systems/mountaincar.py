import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from myriad.systems.base import FiniteHorizonControlSystem


def hill_function(x):
  # return jnp.max(jnp.array([-3 * x - jnp.pi, -1/3 * jnp.cos(3 * x), 3 * x]))
  return x * x


class MountainCar(FiniteHorizonControlSystem):
  def __init__(self, goal_velocity=0):
    """
    Continuous mountain-car, inspired by
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
    """

    self.min_action = -1.0
    self.max_action = 1.0
    self.min_position = -1.2
    self.max_position = 0.6
    self.max_speed = 0.07
    self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    self.goal_velocity = goal_velocity
    self.power = 0.0015
    self.gravity = 0.0025

    super().__init__(
      # [self.np_random.uniform(low=-0.6, high=-0.4), 0]
      x_0=jnp.array([-0.1, 0.]),  # Starting state: position, velocity
      x_T=jnp.array([self.goal_position, self.goal_velocity]),  # Ending state
      T=600.,  # s duration (note, this is not in the original problem)
      bounds=jnp.array([
        [self.min_position, self.max_position],  # Position bounds
        [-self.max_speed, self.max_speed],  # Velocity bounds
        [self.min_action, self.max_action],  # Control bounds
      ]),
      terminal_cost=False,
    )

  # def _height(self, xs):
  #   return jnp.sin(3 * xs) * .45 + .55

  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    # TODO: there is an error somewhere where the calculated state can go outside the boundaries
    # print("input", x_t)
    position, velocity = x_t
    position = jnp.clip(position, a_min=self.min_position, a_max=self.max_position)
    velocity = jnp.clip(velocity, a_min=-self.max_speed, a_max=self.max_speed)
    force = jnp.clip(u_t, a_min=self.min_action, a_max=self.max_action)

    d_position = velocity.squeeze()
    d_velocity = (force * self.power - self.gravity * jax.grad(hill_function)(position)).squeeze()

    # Add in another velocity term that pushes towards the middle if we're too far to the left or right
    # if position <= self.min_position:
    #   d_position = 0.1
    #   d_velocity = 0.
    # elif position >= self.max_position:
    #   d_position = -0.1
    #   d_velocity = 0.

    # d_velocity = force * self.power -self.gravity * jnp.sin(position)

    # print("separate output", d_position, d_velocity)
    # print("output", jnp.array([d_position, d_velocity]))
    return jnp.array([d_position, d_velocity])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return 0.1 * u_t ** 2

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


if __name__ == "__main__":
  import numpy as np
  import matplotlib.pyplot as plt

  mc = MountainCar()

  y1 = np.linspace(-1.5, 1.5, 20)
  y2 = np.linspace(-0.1, 0.1, 20)

  Y1, Y2 = np.meshgrid(y1, y2)

  t = 0

  u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

  NI, NJ = Y1.shape

  for i in range(NI):
    for j in range(NJ):
      x = Y1[i, j]
      y = Y2[i, j]
      yprime = mc.dynamics(jnp.array([x, y]), 0, t)
      u[i, j] = yprime[0]
      v[i, j] = yprime[1]

  Q = plt.quiver(Y1, Y2, u, v, color='r')

  plt.xlabel('position')
  plt.ylabel('velocity')
  plt.xlim([-1.5, 1])
  plt.ylim([-.1, .1])
  plt.show()
