# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp

from typing import Optional

from myriad.custom_types import Control, Cost, DState, Params, State, Timestep
from myriad.systems import FiniteHorizonControlSystem


class RocketLanding(FiniteHorizonControlSystem):
  """
  Simulate a starship landing! Inspired by Thomas Godden's [medium post](https://thomas-godden.medium.com/how-spacex-lands-starship-sort-of-ee96cdde650b).

  This environment models a rocket trying to land vertically on a flat surface, in a similar fashion to how [SpaceX are
  landing their reusable rockets](https://tenor.com/search/rocket-landing-gifs). Usually, the rocket is free-falling
  from an initial horizontal position and must uses its thruster ( \\(u_0(t), u_1(t)\\) ) to both rotate the craft and
  slow down the fall. The goal is to achieve the desired end state while minimizing the fuel usage (minimizing thrust)
  and the angular velocity in order to limit the strain on the vehicle.

  A simplified version of this task form can be modeled as:

  .. math::

    \\begin{align}
    & \\min_{u} \\quad && \\int_0^T u_0(t)^2 + u_1(t)^2 + \\phi'(t)^2 dt \\\\
    & \\; \\mathrm{s.t.}\\quad && x_0''(t) = \\frac{F_v * u_0(t) * \\sin(u_1(t) + \\phi)}{m} \\\\
    & && x_1''(t) = \\frac{F_v * u_0(t) * \\cos(u_1(t) + \\phi)}{m} - g \\\\
    & && \\phi''(t) = \\frac{-6}{F_v * u_0(t) * \\sin(u_1(t)) * m * l} \\\\
    & && x_0(0) = x_0'(0) = 0 ,\\; x_1(0) = h_i ,\\; x_1'(0) = v_i  ,\\; \\phi(0) = -\\pi/2 ,\\; \\phi'(0)=0\\\\
    & && x_0(T) = x_0'(T) = x_1(T) = x_1'(T) = \\phi(T) = \\phi'(T) = 0 \\\\
    & && -1 <= u_0(t) <= 1 \\\\
    & && -F_g <= u_1(t) <= F_g
    \\end{align}

  Notes
  -----
  \\(x_0\\): Horizontal position of the rocket \n
  \\(x_0'\\): Horizontal velocity of the rocket \n
  \\(x_1\\): Height of the rocket \n
  \\(x_1'\\): Falling velocity of the rocket \n
  \\(\\phi\\): Angle of the rocket \n
  \\(\\phi'\\): Angular velocity of the rocket \n
  \\(u_0\\): The vertical thrust, as a ratio of the maximal thrust \\(F_v\\) \n
  \\(u_1\\): The [gimbaled thrust](https://en.wikipedia.org/wiki/Gimbaled_thrust) \n
  \\(F_v\\): Maximal thrust \n
  \\(F_g\\): Maximal gimbaled thrust \n
  \\(g\\): Gravity force \n
  \\(m\\): Total mass of the rocket \n
  \\(l\\): Length of the rocket \n
  \\(h_i, v_i\\): Initial height and falling speed \n
  \\(T\\): The horizon
  """
  # TODO: think about this http://larsblackmore.com/losslessconvexification.htm
  def __init__(self, g: float = 9.8, m: float = 100_000, length: float = 50, width: float = 10) -> None:
    self.g = g  # m/s^2
    self.m = m  # kg
    self.length = length  # m
    self.width = width  # m

    self.min_thrust = 880 * 1000  # N
    self.max_thrust = 1 * 2210 * 1000  # kN

    # Inertia for a uniform density rod
    self.I = 1 / 12 * m * length ** 2

    deg_to_rad = 0.01745329

    self.max_gimble = 20 * deg_to_rad
    self.min_gimble = -self.max_gimble
    self.min_percent_thrust = 0.4
    self.max_percent_thrust = 1.

    # x[0] = x position (m)
    # x[1] = x velocity (m/s)
    # x[2] = y position (m)
    # x[3] = y velocity (m/s)
    # x[4] = angle (rad)
    # x[5] = angular velocity (rad/s)

    # u[0] = thrust (percent)
    # u[1] = thrust angle (rad)
    super().__init__(
      x_0=jnp.array([0., 0., 1000., -80., -jnp.pi / 2., 0.]),
      x_T=jnp.array([0., 0., 0., 0., 0., 0.]),
      T=16.,  # Duration of experiment
      bounds=jnp.array([  # Bounds over the states (x_0, x_1, ...) are given first,
        [-250., 150.],  # followed by bounds over controls (u_0, u_1, ...)
        [-250., 150.],
        [0., 1000.],
        [-250., 150.],
        [-2 * jnp.pi, 2 * jnp.pi],
        [-250., 150.],
        [self.min_percent_thrust, self.max_percent_thrust],
        [self.min_gimble, self.max_gimble],
      ]),
      terminal_cost=False,
    )

  def dynamics(self, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> DState:
    theta = x_t[4]

    thrust = u_t[0]
    thrust_angle = u_t[1]

    # Horizontal force
    F_x = self.max_thrust * thrust * jnp.sin(thrust_angle + theta)
    x_dot = x_t[1]
    x_dotdot = F_x / self.m

    # Vertical force
    F_y = self.max_thrust * thrust * jnp.cos(thrust_angle + theta)
    y_dot = x_t[3]
    y_dotdot = F_y / self.m - self.g

    # Torque
    T = -self.length / 2 * self.max_thrust * thrust * jnp.sin(thrust_angle)
    theta_dot = x_t[5]
    theta_dotdot = T / self.I

    return jnp.array([x_dot, x_dotdot, y_dot, y_dotdot, theta_dot, theta_dotdot])

  def cost(self, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> Cost:
    return u_t[0] ** 2 + u_t[1] ** 2 + 2 * x_t[5] ** 2
