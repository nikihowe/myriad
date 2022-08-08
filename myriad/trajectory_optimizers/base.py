# (c) 2021 Nikolaus Howe
from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
  from myriad.config import Config, HParams
  # from myriad.config import
import jax
import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.flatten_util import ravel_pytree
# from ipopt import minimize_ipopt
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from myriad.config import SystemType
from myriad.nlp_solvers import solve
from myriad.systems import FiniteHorizonControlSystem, IndirectFHCS
from myriad.utils import integrate_in_parallel, integrate_time_independent, \
  integrate_time_independent_in_parallel, integrate_fbsm
from myriad.custom_types import Params


@dataclass
class TrajectoryOptimizer(object):
  """
  An abstract class representing an "optimizer" which can find the solution
    (an optimal trajectory) to a given "system", using a direct approach.
  """
  hp: HParams
  """The hyperparameters"""
  cfg: Config
  """Additional hyperparemeters"""
  objective: Callable[[jnp.ndarray], float]
  """Given a sequence of controls and states, calculates how "good" they are"""
  parametrized_objective: Callable[[Params, jnp.ndarray], float]

  constraints: Callable[[jnp.ndarray], jnp.ndarray]
  """Given a sequence of controls and states, calculates the magnitude of violations of dynamics"""
  parametrized_constraints: Callable[[Params, jnp.ndarray], float]

  bounds: jnp.ndarray
  """Bounds for the states and controls"""
  guess: jnp.ndarray
  """An initial guess for the states and controls"""
  unravel: Callable[[jnp.ndarray], Tuple]
  """Use to separate decision variable array into states and controls"""
  require_adj: bool = False
  """Does this trajectory optimizer require adjoint dynamics in order to work?"""

  def __post_init__(self):
    if self.cfg.verbose:
      # print("optimizer type", self._type)
      print("hp opt type", self.hp.optimizer)
      print("hp quadrature rule", self.hp.quadrature_rule)
      # print(f"x_guess.shape = {self.x_guess.shape}")
      # print(f"u_guess.shape = {self.u_guess.shape}")
      print(f"guess.shape = {self.guess.shape}")
      # print(f"x_bounds.shape = {self.x_bounds.shape}")
      # print(f"u_bounds.shape = {self.u_bounds.shape}")
      print(f"bounds.shape = {self.bounds.shape}")

    if self.hp.system == SystemType.INVASIVEPLANT:
      raise NotImplementedError("Discrete systems are not compatible with Trajectory trajectory_optimizers")

  def solve(self) -> Dict[str, jnp.ndarray]:
    opt_inputs = {
      'objective': self.objective,
      'guess': self.guess,
      'constraints': self.constraints,
      'bounds': self.bounds,
      'unravel': self.unravel
    }

    return solve(self.hp, self.cfg, opt_inputs)
    # TODO: fix solve of FBSM

  def solve_with_params(self, params: Params, guess: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    opt_inputs = {
      'objective': (lambda xs_and_us: self.parametrized_objective(params, xs_and_us)),
      'guess': self.guess,
      'constraints': (lambda xs_and_us: self.parametrized_constraints(params, xs_and_us)),
      'bounds': self.bounds,
      'unravel': self.unravel
    }
    
    if guess is not None:
      opt_inputs['guess'] = guess

    return solve(self.hp, self.cfg, opt_inputs)
    # NOTE: I believe FBSM doesn't work here either

  # You can override these if you want to enable end-to-end planning and model learning
  # def parametrized_objective(self, xs_and_us, params):
  #   raise NotImplementedError
    # return self.objective(xs_and_us)

  # def parametrized_constraints(self, xs_and_us, params):
  #   raise NotImplementedError
    # return self.constraints(xs_and_us)

@dataclass
class IndirectMethodOptimizer(object):
  """
  Abstract class for implementing indirect method trajectory_optimizers, i.e. trajectory_optimizers that relies on the Pontryagin's maximum principle
  """

  hp: HParams
  """The collection of hyperparameters for the experiment"""
  cfg: Config
  """Configuration options that should not impact results"""
  bounds: jnp.ndarray
  """Bounds (lower, upper) over the state variables, followed by the bounds over the controls"""
  guess: jnp.ndarray  # Initial guess on x_t, u_t and adj_t
  """Initial guess for the state, control and adjoint variables"""
  unravel: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
  """Callable to unravel the pytree -- separate decision variable array into states and controls"""
  require_adj: bool = True
  """(bool, optional) -- Does this trajectory optimizer require adjoint dynamics in order to work?"""

  def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve method"""
    raise NotImplementedError

  def stopping_criterion(self, x_iter: Tuple[jnp.ndarray, jnp.ndarray], u_iter: Tuple[jnp.ndarray, jnp.ndarray],
                         adj_iter: Tuple[jnp.ndarray, jnp.ndarray], delta: float = 0.001) -> bool:
    """
    Criterion for stopping the optimization iterations.
    """
    x, old_x = x_iter
    u, old_u = u_iter
    adj, old_adj = adj_iter

    stop_x = jnp.abs(x).sum(axis=0) * delta - jnp.abs(x - old_x).sum(axis=0)
    stop_u = jnp.abs(u).sum(axis=0) * delta - jnp.abs(u - old_u).sum(axis=0)
    stop_adj = jnp.abs(adj).sum(axis=0) * delta - jnp.abs(adj - old_adj).sum(axis=0)

    return jnp.min(jnp.hstack((stop_u, stop_x, stop_adj))) < 0
