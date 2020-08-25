from dataclasses import dataclass
import time
from typing import Callable, Tuple

from jax import grad, jacrev, jit, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import numpy as onp
from ipopt import minimize_ipopt as minimize

from .config import Config, HParams, OptimizerType
from .systems import FiniteHorizonControlSystem
from .utils import integrate


@dataclass
class TrajectoryOptimizer(object):
  hp: HParams
  cfg: Config
  objective: Callable[[np.ndarray], float]
  constraints: Callable[[np.ndarray], np.ndarray]
  bounds: np.ndarray
  guess: np.ndarray
  unravel: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

  def __post_init__(self):
    if self.cfg.verbose:
      print(f"x_guess.shape = {self.x_guess.shape}")
      print(f"u_guess.shape = {self.u_guess.shape}")
      print(f"guess.shape = {self.guess.shape}")
      print(f"x_bounds.shape = {self.x_bounds.shape}")
      print(f"u_bounds.shape = {self.u_bounds.shape}")
      print(f"bounds.shape = {self.bounds.shape}")

  def solve(self) -> Tuple[np.ndarray, np.ndarray]:
    _t1 = time.time()
    solution = minimize(
      fun=jit(self.objective) if self.cfg.jit else self.objective,
      x0=self.guess,
      method='SLSQP',
      constraints=[{
        'type': 'eq',
        'fun': jit(self.constraints) if self.cfg.jit else self.constraints,
        'jac': jit(jacrev(self.constraints)) if self.cfg.jit else jacrev(self.constraints),
      }],
      bounds=self.bounds,
      jac=jit(grad(self.objective)) if self.cfg.jit else grad(self.objective),
      options={
        'max_iter': self.hp.ipopt_max_iter,
      }
    )
    _t2 = time.time()
    if self.cfg.verbose:
      print(f'Solved in {_t2 - _t1} seconds.')

    x, u = self.unravel(solution.x)
    return x, u


def get_optimizer(hp: HParams, cfg: Config, system: FiniteHorizonControlSystem) -> TrajectoryOptimizer:
  if hp.optimizer == OptimizerType.COLLOCATION:
    optimizer = TrapezoidalCollocationOptimizer(hp, cfg, system)
  elif hp.optimizer == OptimizerType.SHOOTING:
    optimizer = MultipleShootingOptimizer(hp, cfg, system)
  else:
    raise KeyError
  return optimizer


class TrapezoidalCollocationOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    N = hp.intervals # Segments
    h = system.T / N # Segment length

    u_guess = np.zeros((N+1,1)) + system.bounds[-1].mean()
    if system.x_T is not None:
      x_guess = np.linspace(system.x_0, system.x_T, num=N+1)
    else:
      _, x_guess = integrate(system.dynamics, system.x_0, u_guess, h, N)
    guess, unravel = ravel_pytree((x_guess, u_guess))
    self.x_guess, self.u_guess = x_guess, u_guess

    def objective(variables: np.ndarray) -> float:
      def fn(x_t1: np.ndarray, x_t2: np.ndarray, u_t1: float, u_t2: float) -> float:
        return (h/2) * (system.cost(x_t1, u_t1) + system.cost(x_t2, u_t2))
      x, u = unravel(variables)
      if system.terminal_cost:
        return system.cost(x[-1], u[-1])
      else:
        return np.sum(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:]))

    def constraints(variables: np.ndarray) -> np.ndarray:
      def fn(x_t1: np.ndarray, x_t2: np.ndarray, u_t1: float, u_t2: float) -> np.ndarray:
        left = (h/2) * (system.dynamics(x_t1, u_t1) + system.dynamics(x_t2, u_t2))
        right = x_t2 - x_t1
        return left - right
      x, u = unravel(variables)
      return np.ravel(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:]))
    
    x_bounds = onp.empty((N+1,system.bounds.shape[0]-1,2))
    x_bounds[:,:,:] = system.bounds[:-1]
    x_bounds[0,:,:] = onp.expand_dims(system.x_0, 1)
    if system.x_T is not None:
      x_bounds[-1,:,:] = onp.expand_dims(system.x_T, 1)
    x_bounds = x_bounds.reshape((-1,2))
    u_bounds = onp.empty((N+1, 2))
    u_bounds[:] = system.bounds[-1:]
    bounds = np.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(hp, cfg, objective, constraints, bounds, guess, unravel)


class MultipleShootingOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    N_x = hp.intervals
    N_u = hp.intervals * hp.controls_per_interval
    h_x = system.T / N_x
    h_u = system.T / N_u

    u_guess = np.zeros((N_u,1)) + system.bounds[-1].mean()
    if system.x_T is not None:
      x_guess = np.linspace(system.x_0, system.x_T, num=N_x+1)[:-1]
    else:
      x_guess = integrate(system.dynamics, system.x_0, u_guess[::hp.controls_per_interval], h_x, N_x)[1][:-1]
    guess, unravel = ravel_pytree((x_guess, u_guess))
    self.x_guess, self.u_guess = x_guess, u_guess

    def objective(variables: np.ndarray) -> float:
      _, u = unravel(variables)
      _, x = integrate(system.dynamics, system.x_0, u, h_u, N_u)
      x = x[1:]
      if system.terminal_cost:
        return system.cost(x[-1], u[-1])
      else:
        return h_u * np.sum(vmap(system.cost)(x, u))
    
    def constraints(variables: np.ndarray) -> np.ndarray:
      x, u = unravel(variables)
      u = u.reshape(hp.intervals, hp.controls_per_interval)
      px, _ = vmap(integrate, in_axes=(None, 0, 0, None, None))(system.dynamics, x, u, h_u, hp.controls_per_interval)
      if system.x_T is not None:
        ex = np.concatenate((x[1:], system.x_T[np.newaxis]))
      else:
        ex = x[1:]
        px = px[:-1]
      return np.ravel(px - ex)

    x_bounds = onp.empty((hp.intervals, system.bounds.shape[0]-1, 2))
    x_bounds[:,:,:] = system.bounds[:-1]
    x_bounds[0,:,:] = np.expand_dims(system.x_0, 1)
    x_bounds = x_bounds.reshape((-1,2))
    u_bounds = onp.empty((hp.intervals * hp.controls_per_interval, 2))
    u_bounds[:] = system.bounds[-1:]
    bounds = np.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(hp, cfg, objective, constraints, bounds, guess, unravel)
