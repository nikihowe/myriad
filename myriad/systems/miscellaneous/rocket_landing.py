# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp

from typing import Optional

from myriad.custom_types import Control, Cost, DState, Params, State, Timestep
from myriad.systems import FiniteHorizonControlSystem


class RocketLanding(FiniteHorizonControlSystem):
  # TODO: think about this http://larsblackmore.com/losslessconvexification.htm
  def __init__(self, g: float = 9.8, m: float = 100_000, length: float = 50, width: float = 10) -> None:
    """
    Simulate a starship landing!
    Inspired by https://thomas-godden.medium.com/how-spacex-lands-starship-sort-of-ee96cdde650b
    """
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
