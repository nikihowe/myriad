from dataclasses import dataclass
import time
from typing import Callable, Tuple

from jax import grad, jacrev, jit, vmap
from jax.flatten_util import ravel_pytree
from jax.ops import index_update
import jax.numpy as np
import numpy as onp
from ipopt import minimize_ipopt as minimize

from .config import Config, HParams, OptimizerType
from .systems import FiniteHorizonControlSystem
from .utils import integrate, integrate_v2


@dataclass
class TrajectoryOptimizer(object):
  hp: HParams
  cfg: Config
  objective: Callable[[np.ndarray], float]
  constraints: Callable[[np.ndarray], np.ndarray]
  bounds: np.ndarray
  guess: np.ndarray
  unravel: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
  require_adj: bool = False

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
  elif hp.optimizer == OptimizerType.FBSM:
    optimizer = FBSM(hp, cfg, system)
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


@dataclass
class IndirectMethodOptimizer(object):
  hp: HParams
  cfg: Config
  bounds: np.ndarray   # Possible bounds on x_t and u_t
  guess: np.ndarray    # Initial guess on x_t, u_t and adj_t
  unravel: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
  require_adj: bool = True


  def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  #return (optimal_state, optimal_control, optimal_adj)
    raise NotImplementedError

  def stopping_criterion(self, x_iter: Tuple[np.ndarray, np.ndarray], u_iter: Tuple[np.ndarray, np.ndarray], adj_iter: Tuple[np.ndarray, np.ndarray], delta: float = 0.001) -> bool:
    x, old_x = x_iter
    u, old_u = u_iter
    adj, old_adj = adj_iter

    stop_x = np.abs(x).sum(axis=0) * delta - np.abs(x - old_x).sum(axis=0)
    stop_u = np.abs(u).sum(axis=0)*delta - np.abs(u-old_u).sum(axis=0)
    stop_adj = np.abs(adj).sum(axis=0) * delta - np.abs(adj - old_adj).sum(axis=0)

    return np.min(np.hstack((stop_u, stop_x, stop_adj))) < 0


class FBSM(IndirectMethodOptimizer):  # Forward-Backward Sweep Method
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    self.system = system
    self.N = hp.steps
    self.h = system.T / self.N
    if system.discrete:
      self.N = system.T
      self.h = 1
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    x_guess = np.vstack((system.x_0, np.zeros((self.N, state_shape))))
    if system.discrete:
      u_guess = np.zeros((self.N, control_shape))
    else:
      u_guess = np.zeros((self.N+1, control_shape))
    if system.adj_T is not None:
      adj_guess = np.vstack((np.zeros((self.N, state_shape)), system.adj_T))
    else :
      adj_guess = np.zeros((self.N+1, state_shape))
    self.t_interval = np.linspace(0, system.T, num=self.N+1).reshape(-1, 1)

    guess, unravel = ravel_pytree((x_guess, u_guess, adj_guess))
    self.x_guess, self.u_guess, self.adj_guess = x_guess, u_guess, adj_guess

    x_bounds = system.bounds[:-1]
    u_bounds = system.bounds[-1:]
    bounds = np.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    #Additional condition if terminal condition are present
    self.terminal_cdtion = False
    if self.system.x_T is not None:
      num_term_state = 0
      for idx, x_Ti in enumerate(self.system.x_T):
        if x_Ti is not None:
          self.terminal_cdtion = True
          self.term_cdtion_state = idx
          self.term_value = x_Ti
          num_term_state += 1
        if num_term_state > 1:
          raise NotImplementedError("Multiple states with terminal condition not supported yet")

    super().__init__(hp, cfg, bounds, guess, unravel)

  def reinitiate(self, a):
    state_shape = self.system.x_0.shape[0]
    control_shape = self.system.bounds.shape[0] - state_shape

    self.x_guess = np.vstack((self.system.x_0, np.zeros((self.N, state_shape))))
    self.u_guess = np.zeros((self.N + 1, control_shape))
    if self.system.adj_T is not None:
      adj_guess = np.vstack((np.zeros((self.N, state_shape)), self.system.adj_T))
    else:
      adj_guess = np.zeros((self.N + 1, state_shape))
    self.adj_guess = index_update(adj_guess, (-1, self.term_cdtion_state), a)

  def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if self.terminal_cdtion:
      return self.sequenceSolver()
    n = 0
    while n==0 or self.stopping_criterion((self.x_guess, old_x), (self.u_guess, old_u), (self.adj_guess, old_adj)):
      old_u = self.u_guess.copy()
      old_x = self.x_guess.copy()
      old_adj = self.adj_guess.copy()

      self.x_guess = integrate_v2(self.system.dynamics, self.x_guess[0], self.u_guess, self.h, self.N, t=self.t_interval, discrete=self.system.discrete)[-1]
      self.adj_guess = integrate_v2(self.system.adj_ODE, self.adj_guess[-1], self.x_guess, -1*self.h, self.N, self.u_guess, t=self.t_interval, discrete=self.system.discrete)[-1]

      u_estimate = self.system.optim_characterization(self.adj_guess, self.x_guess, self.t_interval)
      # Use basic convex approximation to update the guess on u
      self.u_guess = 0.5*(u_estimate + old_u)

      n = n + 1

    return self.x_guess, self.u_guess, self.adj_guess

  def sequenceSolver(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    self.terminal_cdtion = False
    iter = 0

    #Adjust lambda to the initial guess
    a = self.system.guess_a
    self.reinitiate(a)
    x_a, _, _ = self.solve()
    Va = x_a[-1, self.term_cdtion_state] - self.term_value
    b = self.system.guess_b
    self.reinitiate(b)
    x_b, _, _ = self.solve()
    Vb = x_b[-1, self.term_cdtion_state] - self.term_value

    while np.abs(Va) > 1e-10:
      if (np.abs(Va) > np.abs(Vb)):
        a, b = b, a
        Va, Vb = Vb, Va

      d = Va*(b-a)/(Vb-Va)
      b = a
      Vb = Va
      a = a - d
      self.reinitiate(a)
      x_a, _, _ = self.solve()
      Va = x_a[-1, self.term_cdtion_state] - self.term_value
      iter += 1

    return self.x_guess, self.u_guess, self.adj_guess