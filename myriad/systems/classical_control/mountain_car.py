# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp

from typing import Optional

from myriad.custom_types import Control, Cost, DState, Params, State, Timestep
from myriad.systems.base import FiniteHorizonControlSystem


def hill_function(x: float) -> float:
  # return jnp.max(jnp.array([-3 * x - jnp.pi, -1/3 * jnp.cos(3 * x), 3 * x]))
  return 0.5 * x * x


class MountainCar(FiniteHorizonControlSystem):
  """
  Continuous Mountain Car environment, inspired by the [OpenAI gym environment](https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py).
  Model was originally described in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf).

  This environment model a unidimensional car located between two hills, while the goal is often to make it to the top
  of one of the hill. Usually, this environment is made challenging by limiting the force ( \\(u(t)\\) ) the car can
  generate, making it unable to climb directly to the desired steep hill top. In this scenario, the solution
  is to first climb the opposite hill in order to generate enough potential energy to make it on top of the desired hill.

  The system can formally be described as:

  .. math::

    \\begin{align}
    & \\min_{u} \\quad && \\int_0^T u(t)^2  dt \\\\
    & \\; \\mathrm{s.t.}\\quad && x'(t) = p u(t) - g h'(x) \\\\
    & && x(0) = x_i ,\\; x'(0) = v_i \\\\
    & && x(T) = x_f ,\\; x'(T) = v_f \\\\
    & && -1 <= u(t) <= 1
    \\end{align}

  Notes
    -----
    \\(x\\): Position of the car \n
    \\(x'\\): Velocity of the car \n
    \\(p\\): Maximal power that the car engine can output \n
    \\(u\\): The force applied to the car, as a fraction of \\(p\\) \n
    \\(g\\): Gravity force \n
    \\(h(x)\\): Function describing the hill landscape \n
    \\(x_i, v_i\\): Initial position and speed \n
    \\(x_f, v_f\\): Goal position and speed \n
    \\(T\\): The horizon




  """

  def __init__(self, power=0.0015, gravity=0.0025) -> None:
    self.min_action = -1.0
    self.max_action = 1.0
    self.min_position = -1.2
    self.max_position = 0.6
    self.max_speed = 0.07
    self.start_position = -0.1
    self.start_velocity = 0.
    self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    self.goal_velocity = 0
    # self.power = 0.0015
    # self.gravity = 0.0025
    self.power = power
    self.gravity = gravity

    super().__init__(
      # [self.np_random.uniform(low=-0.6, high=-0.4), 0]
      x_0=jnp.array([self.start_position, self.start_velocity]),  # Starting state: position, velocity
      x_T=jnp.array([self.goal_position, self.goal_velocity]),  # Ending state
      T=300.,  # s duration (note, this is not in the original problem)
      bounds=jnp.array([
        [self.min_position, self.max_position],  # Position bounds
        [-self.max_speed, self.max_speed],  # Velocity bounds
        [self.min_action, self.max_action],  # Control bounds
      ]),
      terminal_cost=False,
    )

  # def _height(self, xs):
  #   return jnp.sin(3 * xs) * .45 + .55

  def dynamics(self, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> DState:
    position, velocity = x_t
    force = jnp.clip(u_t, a_min=self.min_action, a_max=self.max_action)

    d_position = velocity.squeeze()
    d_velocity = (force * self.power - self.gravity * jax.grad(hill_function)(position)).squeeze()

    return jnp.array([d_position, d_velocity])

  def parametrized_dynamics(self, params: Params, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> DState:
    position, velocity = x_t
    power = params['power']
    gravity = params['gravity']
    force = jnp.clip(u_t, a_min=self.min_action, a_max=self.max_action)

    d_position = velocity.squeeze()
    d_velocity = (force * power - gravity * jax.grad(hill_function)(position)).squeeze()

    return jnp.array([d_position, d_velocity])

  def cost(self, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> Cost:
    return 10. * u_t ** 2

  def parametrized_cost(self, params: Params, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> Cost:
    return self.cost(x_t, u_t, t)

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
