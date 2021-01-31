from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp

from source.config import SystemType


@dataclass
class FiniteHorizonControlSystem(object):
  _type: SystemType
  x_0: jnp.ndarray  # state at time 0
  x_T: Optional[jnp.ndarray]  # state at time T
  T: float  # duration of trajectory
  bounds: jnp.ndarray  # State and control bounds
  terminal_cost: bool  # Whether only the final state and control are inputs to the cost
  discrete: bool = False  # Whether we are working with a system with continuous cost or not

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError
  
  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[Union[float, jnp.ndarray]]) -> float:
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: Union[float, jnp.ndarray], T: Optional[Union[float, jnp.ndarray]] = None) -> float:
    return 0

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    raise NotImplementedError


@dataclass
class IndirectFHCS(FiniteHorizonControlSystem, ABC):
  adj_T: Optional[jnp.ndarray] = None  # adjoint at time T
  guess_a: Optional[float] = None  # Initial guess for secant method
  guess_b: Optional[float] = None  # Initial guess for secant method

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
    raise NotImplementedError
