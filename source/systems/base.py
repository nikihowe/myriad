from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp


@dataclass
class FiniteHorizonControlSystem(object):
  """
  Abstract class describing a finite-horizon control system.

  :param x_0: The start state
  :param x_T: The end state (optional)
  :param T: The end time (start time is always set to 0)
  :param bounds: Bounds on the states and controls
  :param terminal_cost: Whether or not there is an additional cost added at the end of the trajectory
  :param discrete: Whether or not the system is discrete
  """
  x_0: jnp.ndarray
  x_T: Optional[jnp.ndarray]
  T: float
  bounds: jnp.ndarray
  terminal_cost: bool
  discrete: bool = False

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray]) -> jnp.ndarray:
    #TODO: nh: make this accept time-dependent dynamics
    """
    System dynamics

     ..math::

        f(x(t), u(t), t) = \dot x

    :param x_t: State
    :param u_t: Control
    :return: \dot x
    """
    raise NotImplementedError
  
  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[Union[float, jnp.ndarray]]) -> float:
    """
    Instantaneous cost function

    :param x_t: State
    :param u_t: Control
    :param t: Time
    :return: Instantaneous cost
    """
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: Union[float, jnp.ndarray],
                       T: Optional[Union[float, jnp.ndarray]] = None) -> float:
    """
    Terminal cost function

    :param x_T: Final state
    :param u_T: Final control
    :param T: Final time
    :return: Terminal cost
    """
    return 0

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    """
    Plot your state and control trajectory.

    :param x: State array
    :param u: Control array
    :return:
    """
    raise NotImplementedError


@dataclass
class IndirectFHCS(FiniteHorizonControlSystem, ABC):
  # TODO: @Simon could you please fill out docstrings here?
  adj_T: Optional[jnp.ndarray] = None  # adjoint at time T
  guess_a: Optional[float] = None  # Initial guess for secant method
  guess_b: Optional[float] = None  # Initial guess for secant method

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
  #   raise NotImplementedError
