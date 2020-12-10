from dataclasses import dataclass
import time
from typing import Callable, Tuple, Union, Optional

from jax import grad, jacrev, jit, vmap
from jax.flatten_util import ravel_pytree
from jax.ops import index_update
import jax.numpy as jnp
import numpy as np
from ipopt import minimize_ipopt as minimize

from .config import Config, HParams, OptimizerType, SystemType, NLPSolverType
from .systems import FiniteHorizonControlSystem, IndirectFHCS
from .utils import integrate, integrate_in_parallel, integrate_v2
from .nlp_solvers import extra_gradient


@dataclass
class TrajectoryOptimizer(object):
  _type: OptimizerType
  hp: HParams
  cfg: Config
  objective: Callable[[jnp.ndarray], float]
  constraints: Callable[[jnp.ndarray], jnp.ndarray]
  bounds: jnp.ndarray
  guess: jnp.ndarray
  unravel: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
  require_adj: bool = False

  def __post_init__(self):
    if self.cfg.verbose:
      print(f"x_guess.shape = {self.x_guess.shape}")
      print(f"u_guess.shape = {self.u_guess.shape}")
      print(f"guess.shape = {self.guess.shape}")
      print(f"x_bounds.shape = {self.x_bounds.shape}")
      print(f"u_bounds.shape = {self.u_bounds.shape}")
      print(f"bounds.shape = {self.bounds.shape}")

    if self.hp.system == SystemType.INVASIVEPLANT:
      raise NotImplementedError("Discrete systems are not compatible with Trajectory optimizers")

  def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    _t1 = time.time()
    if self.hp.nlpsolver == NLPSolverType.EXTRAGRADIENT:
      solution = extra_gradient(
        fun=jit(self.objective) if self.cfg.jit else self.objective,
        x0=self.guess,
        method='SLSQP',
        constraints=({
          'type': 'eq',
          'fun': jit(self.constraints) if self.cfg.jit else self.constraints,
          'jac': jit(jacrev(self.constraints)) if self.cfg.jit else jacrev(self.constraints),
        }),
        bounds=self.bounds,
        jac=jit(grad(self.objective)) if self.cfg.jit else grad(self.objective),
        options={
          'maxiter': self.hp.ipopt_max_iter,
        }
      )
    else:
      solution = minimize(
        fun=jit(self.objective) if self.cfg.jit else self.objective,
        x0=self.guess,
        method='SLSQP',
        constraints=({
          'type': 'eq',
          'fun': jit(self.constraints) if self.cfg.jit else self.constraints,
          'jac': jit(jacrev(self.constraints)) if self.cfg.jit else jacrev(self.constraints),
        }),
        bounds=self.bounds,
        jac=jit(grad(self.objective)) if self.cfg.jit else grad(self.objective),
        options={
          'maxiter': self.hp.ipopt_max_iter,
        }
      )
    _t2 = time.time()
    if self.cfg.verbose:
      print(f'Solved in {_t2 - _t1} seconds.')

    x, u = self.unravel(solution.x)
    return x, u


@dataclass
class IndirectMethodOptimizer(object):
  hp: HParams
  cfg: Config
  bounds: jnp.ndarray   # Possible bounds on x_t and u_t
  guess: jnp.ndarray    # Initial guess on x_t, u_t and adj_t
  unravel: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
  require_adj: bool = True

  def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    raise NotImplementedError

  def stopping_criterion(self, x_iter: Tuple[jnp.ndarray, jnp.ndarray], u_iter: Tuple[jnp.ndarray, jnp.ndarray],
                         adj_iter: Tuple[jnp.ndarray, jnp.ndarray], delta: float = 0.001) -> bool:
    x, old_x = x_iter
    u, old_u = u_iter
    adj, old_adj = adj_iter

    stop_x = jnp.abs(x).sum(axis=0) * delta - jnp.abs(x - old_x).sum(axis=0)
    stop_u = jnp.abs(u).sum(axis=0)*delta - jnp.abs(u-old_u).sum(axis=0)
    stop_adj = jnp.abs(adj).sum(axis=0) * delta - jnp.abs(adj - old_adj).sum(axis=0)

    return jnp.min(jnp.hstack((stop_u, stop_x, stop_adj))) < 0


def get_optimizer(hp: HParams, cfg: Config, system: Union[FiniteHorizonControlSystem, IndirectFHCS]
                  ) -> Union[TrajectoryOptimizer, IndirectMethodOptimizer]:
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
    num_intervals = hp.intervals  # Segments
    h = system.T / num_intervals  # Segment length
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    u_guess = jnp.zeros((num_intervals+1, control_shape))
    u_mean = system.bounds[-1 * control_shape:].mean()

    if (not jnp.isnan(jnp.sum(u_mean))) and (not jnp.isinf(u_mean).any()):  # handle bounds with infinite values
      u_guess += u_mean

    if system.x_T is not None:
      # We need to handle the cases where a terminal bound is specified only for some state variables, not all
      row_guesses = []
      # TODO: make sure that nh correctly understood that the special case which has been commented
      #       out below is properly handled by the expansion of the loop
      # if system.x_T[0] is not None:
      #   x_guess = jnp.linspace(system.x_0[0], system.x_T[0], num=num_intervals + 1).reshape(-1, 1)
      # else: # the first state component has no final constraints
      #   _, x_guess = integrate(system.dynamics, system.x_0, u_guess, h, num_intervals)
      #   x_guess = x_guess[:, 0].reshape(-1, 1)
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=num_intervals+1).reshape(-1, 1)
        else:
          _, row_guess = integrate(system.dynamics, system.x_0, u_guess, h, num_intervals)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else: # no final state requirement
      _, x_guess = integrate(system.dynamics, system.x_0, u_guess, h, num_intervals)
    guess, unravel_decision_variables = ravel_pytree((x_guess, u_guess))
    self.x_guess, self.u_guess = x_guess, u_guess

    def objective(variables: jnp.ndarray) -> float:
      def fn(x_t1: jnp.ndarray, x_t2: jnp.ndarray, u_t1: float, u_t2: float, t1: float, t2: float) -> float:
        return (h/2) * (system.cost(x_t1, u_t1, t1) + system.cost(x_t2, u_t2, t2))
      x, u = unravel_decision_variables(variables)
      t = jnp.linspace(0, system.T, num=num_intervals+1)  # Support cost function with dependency on t
      if system.terminal_cost:
        return jnp.sum(system.terminal_cost_fn(x[-1], u[-1])) + jnp.sum(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:], t[:-1],
                                                                                 t[1:]))
      else:
        return jnp.sum(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:], t[:-1], t[1:]))

    def constraints(variables: jnp.ndarray) -> jnp.ndarray:
      def fn(x_t1: jnp.ndarray, x_t2: jnp.ndarray, u_t1: float, u_t2: float) -> jnp.ndarray:
        left = (h/2) * (system.dynamics(x_t1, u_t1) + system.dynamics(x_t2, u_t2))
        right = x_t2 - x_t1
        return left - right
      x, u = unravel_decision_variables(variables)
      return jnp.ravel(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:]))
    
    x_bounds = np.empty((num_intervals+1, system.bounds.shape[0]-control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]
    x_bounds[0, :, :] = np.expand_dims(system.x_0, 1)
    if system.x_T is not None:
      x_bounds[-control_shape, :, :] = np.expand_dims(system.x_T, 1)
    x_bounds = x_bounds.reshape((-1, 2))
    u_bounds = np.empty(((num_intervals+1)*control_shape, 2))
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape-i)*(num_intervals+1):(control_shape-i+1)*(num_intervals+1)] = system.bounds[-i]
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(OptimizerType.COLLOCATION, hp, cfg, objective, constraints, bounds, guess, unravel_decision_variables)


class MultipleShootingOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    N_x = hp.intervals
    N_u = hp.intervals * hp.controls_per_interval
    h_x = system.T / N_x
    h_u = system.T / N_u
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    u_guess = jnp.zeros((N_u, control_shape))
    u_mean = system.bounds[-1 * control_shape:].mean()
    if (not jnp.isnan(jnp.sum(u_mean))) and (not jnp.isinf(u_mean).any()):  # handle bounds with infinite values
      u_guess += u_mean

    # TODO: have one more control at the final time also

    if system.x_T is not None:
      # We need to handle the cases where a terminal bound is specified only for some state variables, not all
      # if system.x_T[0] is not None:
      #   x_guess = jnp.linspace(system.x_0[0], system.x_T[0], num=N_x+1)[:-1].reshape(-1, 1)
      # else:
      #   x_guess = integrate(system.dynamics, system.x_0, u_guess[::hp.controls_per_interval], h_x, N_x)[1][:-1]
      #   x_guess = x_guess[:, 0].reshape(-1, 1)
      row_guesses = [] # TODO: check if this behaves the same as the earlier code
      # For the state variables which have a required end state, interpolate between start and end;
      # otherwise, use rk4 with initial controls as a first guess at intermediate and end state values
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=N_x+1).reshape(-1, 1)
        else:
          _, row_guess = integrate(system.dynamics, system.x_0, u_guess[::hp.controls_per_interval], h_x, N_x)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else:
      _, x_guess = integrate(system.dynamics, system.x_0, u_guess[::hp.controls_per_interval], h_x, N_x)
    guess, unravel = ravel_pytree((x_guess, u_guess))
    assert len(x_guess) == N_x + 1 # we have one state decision var for each node, including start and end
    self.x_guess, self.u_guess = x_guess, u_guess

    # Augment the dynamics so we can integrate cost the same way we do state
    def augmented_dynamics(x_and_c: jnp.ndarray, u: float) -> jnp.ndarray:
      x, c = x_and_c[:-1], x_and_c[-1]
      return jnp.append(system.dynamics(x, u), system.cost(x, u))

    def objective(variables: jnp.ndarray) -> float:
      # This code runs faster, but only does a linear interpolation for cost.
      # Better to have the interpolation match the integration scheme,
      # and just use Euler / Heun if we need shooting to be faster

      # xs, us = unravel(variables)
      # t = jnp.linspace(0, system.T, num=N_x+1)[:-1]  # Support cost function with dependency on t
      # t = jnp.repeat(t, hp.controls_per_interval)
      # _, x = integrate(system.dynamics, system.x_0, u, h_u, N_u)
      # x = x[:-1]
      # if system.terminal_cost:
      #   return jnp.sum(system.terminal_cost_fn(x[-1], u[-1])) + h_u * jnp.sum(vmap(system.cost)(x, u, t))
      # else:
      #   return h_u * jnp.sum(vmap(system.cost)(x, u, t))
      
      # ---
      xs, us = unravel(variables)
      t = jnp.linspace(0, system.T, num=N_x+1)  # Support cost function with dependency on t
      t = jnp.repeat(t, hp.controls_per_interval)

      starting_xs_and_costs = jnp.hstack([xs[:-1], jnp.zeros(len(xs[:-1])).reshape(-1, 1)])

      # Integrate cost in parallel
      states_and_costs, _ = integrate_in_parallel(
        augmented_dynamics, starting_xs_and_costs, us.reshape(hp.intervals, hp.controls_per_interval),
        h_u, hp.controls_per_interval, None)

      costs = jnp.sum(states_and_costs[:,-1])
      if system.terminal_cost:
        last_augmented_state = states_and_costs[-1]
        costs += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])
      
      return jnp.sum(costs)

    
    def constraints(variables: jnp.ndarray) -> jnp.ndarray:
      xs, us = unravel(variables)
      us = us.reshape(hp.intervals, hp.controls_per_interval, control_shape)
      us = jnp.squeeze(us) # removes the control shape dimension if it's 1
      px, _ = integrate_in_parallel(system.dynamics, xs[:-1], us, h_u, hp.controls_per_interval, None)
      return jnp.ravel(px - xs[1:])

    ############################
    # State and Control Bounds #
    ############################

    # State decision variables at every node
    x_bounds = np.empty((hp.intervals + 1, system.bounds.shape[0] - control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]

    # Starting state
    x_bounds[0, :, :] = jnp.expand_dims(system.x_0, 1)

    # Ending state
    if system.x_T is not None:
      x_bounds[-1, :, :] = jnp.expand_dims(system.x_T, 1)

    x_bounds = x_bounds.reshape((-1, 2))

    # Conrol decision variables at every node, plus at intermediate points
    # TODO: put one more control at the final state
    u_bounds = np.empty((hp.intervals * hp.controls_per_interval*control_shape, 2))
    N = hp.intervals * hp.controls_per_interval
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (N + 1):(control_shape - i + 1) * (N + 1)] = system.bounds[-i]
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(OptimizerType.SHOOTING, hp, cfg, objective, constraints, bounds, guess, unravel)


class FBSM(IndirectMethodOptimizer):  # Forward-Backward Sweep Method
  def __init__(self, hp: HParams, cfg: Config, system: IndirectFHCS):
    self.system = system
    self.N = hp.steps
    self.h = system.T / self.N
    if system.discrete:
      self.N = int(system.T)
      self.h = 1
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    x_guess = jnp.vstack((system.x_0, jnp.zeros((self.N, state_shape))))
    if system.discrete:
      u_guess = jnp.zeros((self.N, control_shape))
    else:
      u_guess = jnp.zeros((self.N+1, control_shape))
    if system.adj_T is not None:
      adj_guess = jnp.vstack((jnp.zeros((self.N, state_shape)), system.adj_T))
    else:
      adj_guess = jnp.zeros((self.N+1, state_shape))
    self.t_interval = jnp.linspace(0, system.T, num=self.N+1).reshape(-1, 1)

    guess, unravel = ravel_pytree((x_guess, u_guess, adj_guess))
    self.x_guess, self.u_guess, self.adj_guess = x_guess, u_guess, adj_guess

    x_bounds = system.bounds[:-1]
    u_bounds = system.bounds[-1:]
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    # Additional condition if terminal condition are present
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

    self.x_guess = jnp.vstack((self.system.x_0, jnp.zeros((self.N, state_shape))))
    self.u_guess = jnp.zeros((self.N + 1, control_shape))
    if self.system.adj_T is not None:
      adj_guess = jnp.vstack((jnp.zeros((self.N, state_shape)), self.system.adj_T))
    else:
      adj_guess = jnp.zeros((self.N + 1, state_shape))
    self.adj_guess = index_update(adj_guess, (-1, self.term_cdtion_state), a)

  def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if self.terminal_cdtion:
      return self.sequencesolver()
    n = 0
    while n == 0 or self.stopping_criterion((self.x_guess, old_x), (self.u_guess, old_u), (self.adj_guess, old_adj)):
      old_u = self.u_guess.copy()
      old_x = self.x_guess.copy()
      old_adj = self.adj_guess.copy()

      self.x_guess = integrate_v2(self.system.dynamics, self.x_guess[0], self.u_guess, self.h, self.N,
                                  t=self.t_interval, discrete=self.system.discrete)[-1]
      self.adj_guess = integrate_v2(self.system.adj_ODE, self.adj_guess[-1], self.x_guess, -1*self.h, self.N,
                                    self.u_guess, t=self.t_interval, discrete=self.system.discrete)[-1]

      u_estimate = self.system.optim_characterization(self.adj_guess, self.x_guess, self.t_interval)
      # Use basic convex approximation to update the guess on u
      self.u_guess = 0.5*(u_estimate + old_u)

      n = n + 1

    return self.x_guess, self.u_guess, self.adj_guess

  def sequencesolver(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    self.terminal_cdtion = False
    count = 0

    # Adjust lambda to the initial guess
    a = self.system.guess_a
    self.reinitiate(a)
    x_a, _, _ = self.solve()
    Va = x_a[-1, self.term_cdtion_state] - self.term_value
    b = self.system.guess_b
    self.reinitiate(b)
    x_b, _, _ = self.solve()
    Vb = x_b[-1, self.term_cdtion_state] - self.term_value

    while jnp.abs(Va) > 1e-10:
      if jnp.abs(Va) > jnp.abs(Vb):
        a, b = b, a
        Va, Vb = Vb, Va

      d = Va*(b-a)/(Vb-Va)
      b = a
      Vb = Va
      a = a - d
      self.reinitiate(a)
      x_a, _, _ = self.solve()
      Va = x_a[-1, self.term_cdtion_state] - self.term_value
      count += 1

    return self.x_guess, self.u_guess, self.adj_guess
