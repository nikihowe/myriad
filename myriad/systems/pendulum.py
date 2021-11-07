# (c) 2021 Nikolaus Howe

# inspired by https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# and https://github.com/locuslab/mpc.pytorch/blob/07f43da67581b783f4f230ca97b0efbc421773af/mpc/env_dx/pendulum.py

import jax
import jax.numpy as jnp

from typing import Optional

from myriad.systems.base import FiniteHorizonControlSystem
from myriad.custom_types import Control, DState, Params, State, Timestep


# https://github.com/openai/gym/blob/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/envs/classic_control/pendulum.py#L101
@jax.jit
def angle_normalize(x):
  return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


class Pendulum(FiniteHorizonControlSystem):
  def __init__(self, g: float = 10., m: float = 1., length: float = 1.):
    # Learnable parameters
    self.g = g
    self.m = m
    self.length = length

    # Fixed parameters
    self.max_speed = 8.
    self.max_torque = 2.
    self.x_0 = jnp.array([0., 0.])
    self.x_T = jnp.array([jnp.pi, 0.])
    self.ctrl_penalty = 0.001

    super().__init__(
      x_0=self.x_0,  # Starting state: position, velocity
      x_T=self.x_T,  # Ending state
      T=15,  # s duration (note, this is not in the original problem)
      bounds=jnp.array([
        [-jnp.pi, jnp.pi],  # theta
        [-self.max_speed, self.max_speed],  # dtheta
        [-self.max_torque, self.max_torque],  # Control bounds
      ]),
      terminal_cost=False,
    )

  def parametrized_dynamics(self, params: Params, x: State, u: Control, t: Optional[Timestep] = None) -> DState:
    u = jnp.clip(u, a_min=-self.max_torque, a_max=self.max_torque)

    g = params['g']
    m = params['m']
    length = params['length']

    theta, dot_theta = x
    theta = angle_normalize(theta)
    dot_theta = jnp.clip(dot_theta, a_min=-self.max_speed, a_max=self.max_speed)
    # print("theta, dot_theta", x)

    dot_dot_theta = (-3. * g / (2. * length) * jnp.sin(theta)
                     + 3. * u / (m * length ** 2)).squeeze() * 0.05

    # print("dot theta", dot_theta)
    # print("dot dot", dot_dot_theta)
    return jnp.array([dot_theta, dot_dot_theta])

  def dynamics(self, x: State, u: Control, t: Optional[Timestep] = None) -> DState:
    u = jnp.clip(u, a_min=-self.max_torque, a_max=self.max_torque)

    theta, dot_theta = x
    theta = angle_normalize(theta)
    dot_theta = jnp.clip(dot_theta, a_min=-self.max_speed, a_max=self.max_speed)
    # print("theta, dot_theta", x)

    dot_dot_theta = (-3. * self.g / (2. * self.length) * jnp.sin(theta)
                     + 3. * u / (self.m * self.length ** 2)).squeeze() * 0.05

    # print("dot theta", dot_theta)
    # print("dot dot", dot_dot_theta)
    return jnp.array([dot_theta, dot_dot_theta])

  def parametrized_cost(self, params: Params, x: State, u: Control, t: Timestep):
    # Do nothing, for now
    return self.cost(x, u, t)

  def cost(self, x: State, u: Control, t: Timestep) -> float:
    # print("state is", x)
    assert len(x) == 2

    theta, dot_theta = x
    return angle_normalize(theta)**2 + 0.1 * dot_theta**2 + self.ctrl_penalty * u**2


if __name__ == "__main__":
  pass

  # import numpy as np
  # import matplotlib.pyplot as plt
  #
  # pd = Pendulum()
  #
  # y1 = np.linspace(-2*jnp.pi, 2*jnp.pi, 20)
  # y2 = np.linspace(-pd.max_speed, pd.max_speed, 20)
  #
  # Y1, Y2 = np.meshgrid(y1, y2)
  #
  # t = 0
  #
  # u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
  #
  # NI, NJ = Y1.shape
  #
  # for i in range(NI):
  #   for j in range(NJ):
  #     x = Y1[i, j]
  #     y = Y2[i, j]
  #     yprime = pd.dynamics(jnp.array([x, y]), 0, t)
  #     u[i, j] = yprime[0]
  #     v[i, j] = yprime[1]
  #
  # Q = plt.quiver(Y1, Y2, u, v, color='r')
  #
  # plt.xlabel('angle')
  # plt.ylabel('angular velocity')
  # plt.xlim([-2*jnp.pi, 2*jnp.pi])
  # plt.ylim([-pd.max_speed, pd.max_speed])
  # plt.show()
  # def get_frame(self, x, ax=None):
  #   x = util.get_data_maybe(x.view(-1))
  #   assert len(x) == 3
  #   l = self.params[2].item()
  #
  #   cos_th, sin_th, dth = torch.unbind(x)
  #   th = np.arctan2(sin_th, cos_th)
  #   x = sin_th * l
  #   y = cos_th * l
  #
  #   if ax is None:
  #     fig, ax = plt.subplots(figsize=(6, 6))
  #   else:
  #     fig = ax.get_figure()
  #
  #   ax.plot((0, x), (0, y), color='k')
  #   ax.set_xlim((-l * 1.2, l * 1.2))
  #   ax.set_ylim((-l * 1.2, l * 1.2))
  #   return fig, ax

  # def get_true_obj(self):
  #   q = torch.cat((
  #     self.goal_weights,
  #     self.ctrl_penalty * torch.ones(self.n_ctrl)
  #   ))
  #   assert not hasattr(self, 'mpc_lin')
  #   px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
  #   p = torch.cat((px, torch.zeros(self.n_ctrl)))
  #   return Variable(q), Variable(p)