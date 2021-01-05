from typing import Tuple
from gym import Env
from gym.spaces import Box
import numpy as np
import jax.numpy as jnp

from .config import IntegrationOrder
from .utils import integrate
from .systems import FiniteHorizonControlSystem


class FiniteHorizonControlEnv(Env):
  def __init__(self, system: FiniteHorizonControlSystem, intervals: int, order: IntegrationOrder):
    super().__init__()
    self.sys = system
    self.order = order
    self.max_i = intervals
    self.ss = self.sys.T / self.max_i # step size
    # Env state
    self.s = self.sys.x_0 # current state
    self.i = 0 # current (completed) interval
    self.s_hist = []
    self.c_hist = []
    # Env attributes
    self.observation_space = Box(np.asarray(self.sys.bounds[:-1,0]), np.asarray(self.sys.bounds[:-1,1]), self.s.shape)
    self.action_space = Box(np.asarray(self.sys.bounds[-1:,0]), np.asarray(self.sys.bounds[-1:,1]))

  @property
  def d(self) -> bool:
    # Done
    return self.i >= self.max_i
  
  @property
  def t(self) -> float:
    # Timestep
    return self.ss * self.i

  def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
    c = jnp.array(action) # Control
    if self.d:
      print('System rollout already completed.')
      r = 0.0
    else:
      prev_s = self.s
      prev_t = self.t
      self.s = integrate(self.sys.dynamics, self.s, c, self.ss, 1, None, self.order)[0]
      self.i += 1
      self.s_hist.append(self.s)
      self.c_hist.append(c)
      
      cost = (self.ss / 2) * (self.sys.cost(prev_s, c, prev_t) + self.sys.cost(self.s, c, self.t))
      if self.d:
        self.c_hist.append(c)
        if self.sys.terminal_cost:
          cost += self.sys.terminal_cost_fn(self.s, c, self.t)
        if self.sys.x_T is not None:
          cost += np.mean((self.s - self.sys.x_T) ** 2) # MSE loss
      r = float(-cost) # Reward
      
    return np.asarray(self.s), r, self.d, {}

  def reset(self) -> np.ndarray:
    self.s = self.sys.x_0
    self.i = 0
    self.s_hist.append(self.s)
    return np.asarray(self.s)
  
  def render(self, mode='human') -> None:
    if self.d:
      self.sys.plot_solution(np.array(self.s_hist), np.array(self.c_hist))
