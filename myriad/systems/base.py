from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp


@dataclass
class FiniteHorizonControlSystem(object):
  """
  Abstract class describing a finite-horizon control system. Model a problem of the form:

  .. math::

    \\begin{align}
    &\\min_u \\quad &&g_T(x_T,u_T,T) + \\int_0^T g(x,u,t) dt \\\\
    & \\; \\mathrm{s.t.}\\quad && x'(t) = f(x,u,t) \\\\
    & && x(0)=x_0
    \\end{align}
  """
  x_0: jnp.ndarray
  """ State at time 0"""
  x_T: Optional[jnp.ndarray]
  """State at time T"""
  T: float
  """Duration of trajectory"""
  bounds: jnp.ndarray
  """State and control bounds"""
  terminal_cost: bool
  """Whether or not there is an additional cost added at the end of the trajectory"""
  discrete: bool = False
  """Whether or not the system is discrete"""

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray]) -> jnp.ndarray:
    # TODO: nh: make this accept time-dependent dynamics
    """ The set of equations defining the dynamics of the system. For continuous system, return the vector fields
     of the state variables \\(x\\) under the influence of the controls \\(u\\), i.e.:
     $$x'(t) = f(x,u,t)$$

    Args:
        x_t: (jnp.ndarray) -- An array, representing the state variables at various time t
        u_t: (float, jnp.ndarray) -- An array, representing the control variables at various time t
    Returns:
        dx_t: (jnp.ndarray) -- The derivative value of the state variables, x_t, at corresponding time t
     """
    raise NotImplementedError

  def cost(self, x_t: Union[float, jnp.ndarray], u_t: Union[float, jnp.ndarray],
           t: Optional[Union[float, jnp.ndarray]]) -> float:
    """ The instantaneous time function that the system seeks to minimize.

    Args:
        x_t: (jnp.ndarray) -- State variables at time t
        u_t: (jnp.ndarray) -- Control variables at time t
        t: (jnp.ndarray, optional) -- Time parameter
    Returns:
        cost: (float) -- The instantaneous cost \\( g(x_t,u_t,t) \\)
    """
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: Union[float, jnp.ndarray], u_T: Union[float, jnp.ndarray],
                       T: Optional[Union[float, jnp.ndarray]] = None) -> float:
    """ The cost function associated to the final state

    Args:
        x_T: (jnp.ndarray) -- Final state
        u_T: (jnp.ndarray) -- Final control
        T: (jnp.ndarray, float) -- The Horizon
    Returns:
        cost_T: (float) -- The terminal cost \\(g_T(x_T,u_T,T\\)
    """
    return 0

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    """ The plotting tool for the current system

    Args:
      x: State array
      u: Control array
    """

    raise NotImplementedError


@dataclass
class IndirectFHCS(FiniteHorizonControlSystem, ABC):
  """
    Augment the base class for defining control problem under a finite horizon so that indirect methods can be use.
    Model a problem of the form:

    .. math::

      \\begin{align}
      & \\min_u \\quad && g_T(x_T,u_T,T) + \\int_0^T g(x,u,t) dt\\\\
      & \\; \\mathrm{s.t.}\\quad && x'(t) = f(x,u,t)\\\\
      & &&x(0)=x_0
      \\end{align}

    Taking into account the adjoint dynamics and the optimal characterization given by the Pontryagin's maximum principle
    """
  adj_T: Optional[jnp.ndarray] = None
  """Adjoint at time T"""
  guess_a: Optional[float] = None
  """Initial lower guess for secant method"""
  guess_b: Optional[float] = None
  """Initial upper guess for secant method"""

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    The adjoint dynamics, given by:
    $$\\lambda '(t) = -\\frac{\\partial H}{\\partial x}$$

    \\( H \\) being the system Hamiltonian

    Args:
        adj_t: (jnp.ndarray) -- An array, representing the adjoint variables at various time t
        x_t: (jnp.ndarray) -- An array, representing the state variables at various time t
        u_t: (jnp.ndarray, optional) -- An array, representing the control variables at various time t
        t: (jnp.ndarray, optional) -- The time array, for time-dependent systems
    Returns:
        d_adj_t: (jnp.ndarray) -- The derivative value of the adjoint variables, \\(\\lambda\\), at corresponding time t
    """
    raise NotImplementedError

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    The optimality characterization of the controls w/r to the state and adjoint variables. That is, the controls cannot
    be optimal if they don't satisfy:
    $$\\frac{\\partial H}{\\partial u} = 0 \\; \\mathrm{at} \\; u^*$$
    This leads to the following condition, the optimality characterization, on \\(u^*\\) if \\(H\\) is quadratic in
     \\(u\\):
    $$u^* = h(x,t)$$

    Args:
        adj_t: (jnp.ndarray) -- An array, representing the adjoint variables at various time t
        x_t: (jnp.ndarray, optional) -- An array, representing the state variables at various time t
        t: (jnp.ndarray, optional) -- The time array, for time-dependent systems
    Returns:
        u_star: (jnp.ndarray) -- Control candidates at corresponding time t that meets the above condition
    """
    raise NotImplementedError
