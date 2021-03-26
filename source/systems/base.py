from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp


@dataclass
class FiniteHorizonControlSystem(object):
  """
  Base class for defining control problem under a finite horizon. Model a problem of the form:
  $$\\min_u \\quad g_T(x_T,u_T,T) + \int_0^T g(x,u,t) dt$$
  $$\mathrm{s.t.}\qquad \dot{x} = f(x,u,t)$$
  $$x(0)=x_0$$
  """
  x_0: jnp.ndarray
  """ state at time 0"""
  x_T: Optional[jnp.ndarray]
  """state at time T"""
  T: float
  """duration of trajectory"""
  bounds: jnp.ndarray
  """State and control bounds"""
  terminal_cost: bool
  """Whether only the final state and control are inputs to the cost"""
  discrete: bool = False
  """Whether we are working with a system with continuous cost or not"""

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """ The set of equations defining the dynamics of the system. In continuous system, return the vector fields
     of the state variables \\(x\\) under the influence of the controls \\(u\\), i.e.:
     $$\\dot{x} = f(x,u,t)$$

    Args:
        x_t: state variables at time t
        u_t: control variables at time t
    Returns:
        A jax.numpy.ndarray, representing the derivative value of the state variables, x_t, at time t
     """
    raise NotImplementedError
  
  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[Union[float, jnp.ndarray]]) -> float:
    """ The continuous time function that the system seeks to minimize.

    Args:
        x_t: state variables at time t
        u_t: control variables at time t
        t: time parameter
    Returns:
        the instantaneous cost \\( g(x_t,u_t,t) \\)
    """
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: Union[float, jnp.ndarray], T: Optional[Union[float, jnp.ndarray]] = None) -> float:
    """ The cost function associated to the final state

    Args:
        x_T: final state variables
        u_T: final control variables
        T: horizon
    Returns:
        the terminal cost \\(g_T(x_T,u_T,T\\)
    """
    return 0

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    """ The plotting tool for the current system"""
    raise NotImplementedError


@dataclass
class IndirectFHCS(FiniteHorizonControlSystem, ABC):
  """
    Augment the base class for defining control problem under a finite horizon so that indirect methods can be use.
    Model a problem of the form:
    $$\\min_u \\quad g_T(x_T,u_T,T) + \int_0^T g(x,u,t) dt$$
    $$\mathrm{s.t.}\qquad \dot{x} = f(x,u,t)$$
    $$x(0)=x_0$$

    Taking into account the adjoint dynamics and the optimal characterization given by the Pontryagin's maximum principle
    """
  adj_T: Optional[jnp.ndarray] = None
  """adjoint at time T"""
  guess_a: Optional[float] = None
  """initial lower guess for secant method"""
  guess_b: Optional[float] = None
  """initial upper guess for secant method"""

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    The adjoint dynamics, given by:
    $$\\dot{\\lambda} = -\\frac{\\partial H}{\\partial x}$$

    \\( H \\) being the system Hamiltonian

    Args:
        adj_t: adjoint variables at time t
        x_t: state variables at time t
        u_t: control variables at time t
        t: time parameter
    Returns:
        A jax.numpy.ndarray, representing the derivative value of the adjoint variables, \\(\\lambda\\), at time t
    """
    raise NotImplementedError

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    The optimality characterization of the controls w/r to the state and adjoint variables. That is, the controls cannot
    be optimal if they don't satisfy:
    $$\\frac{\\partial H}{\\partial u} = 0 \\; \\mathrm{at} \\; u^*$$

    Args:
        adj_t: adjoint variables at time t
        x_t: state variables at time t
        t: time parameter
    Returns:
        A jnp.ndarray, containing controls candidate at time t that meets the above condition
    """
    raise NotImplementedError

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
  #   raise NotImplementedError
