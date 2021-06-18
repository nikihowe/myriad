from dataclasses import dataclass
import time
from typing import Callable, Tuple, Union, Optional

import jax
from jax import vmap
import functools
from jax.flatten_util import ravel_pytree
from jax.ops import index_update
import jax.numpy as jnp
import numpy as np
# from ipopt import minimize_ipopt
from scipy.optimize import minimize
from typing import Dict

from myriad.config import Config, HParams, OptimizerType, SystemType, NLPSolverType, IntegrationOrder
from myriad.systems import FiniteHorizonControlSystem, IndirectFHCS
from myriad.utils import integrate, integrate_in_parallel, integrate_time_independent_in_parallel, integrate_fbsm, solve


@dataclass
class TrajectoryOptimizer(object):
  """
  An abstract class representing an "optimizer" which can find the solution
    (an optimal trajectory) to a given "system", using a direct approach.
  """
  _type: OptimizerType  # TODO: fix this awkward typing system
  """The kind of optimizer this is (options defined in config)"""
  hp: HParams
  """The hyperparameters"""
  cfg: Config
  """Additional hyperparemeters"""
  objective: Callable[[jnp.ndarray], float]
  """Given a sequence of controls and states, calculates how "good" they are"""
  constraints: Callable[[jnp.ndarray], jnp.ndarray]
  """Given a sequence of controls and states, calculates the magnitude of violations of dynamics"""
  bounds: jnp.ndarray
  """Bounds for the states and controls"""
  guess: jnp.ndarray
  """An initial guess for the states and controls"""
  unravel: Callable[[jnp.ndarray], Tuple]
  """Use to separate decision variable array into states and controls"""
  require_adj: bool = False
  """Does this trajectory optimizer require adjoint dynamics in order to work?"""

  def __post_init__(self):
    # if self.cfg.verbose:
    #   print(f"x_guess.shape = {self.x_guess.shape}")
    #   print(f"u_guess.shape = {self.u_guess.shape}")
    #   print(f"guess.shape = {self.guess.shape}")
    #   print(f"x_bounds.shape = {self.x_bounds.shape}")
    #   print(f"u_bounds.shape = {self.u_bounds.shape}")
    #   print(f"bounds.shape = {self.bounds.shape}")

    if self.hp.system == SystemType.INVASIVEPLANT:
      raise NotImplementedError("Discrete systems are not compatible with Trajectory optimizers")

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


@dataclass
class IndirectMethodOptimizer(object):
  """
  Abstract class for implementing indirect method optimizers, i.e. optimizers that relies on the Pontryagin's maximum principle
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


def get_optimizer(hp: HParams, cfg: Config, system: Union[FiniteHorizonControlSystem, IndirectFHCS]
                  ) -> Union[TrajectoryOptimizer, IndirectMethodOptimizer]:
  """ Helper function to fetch the desired optimizer for system resolution"""
  if hp.optimizer == OptimizerType.COLLOCATION:
    if hp.order == IntegrationOrder.CONSTANT:
      print("CONSTANT collocation not yet implemented; using LINEAR collocation for now")
      optimizer = TrapezoidalCollocationOptimizer(hp, cfg, system)
    elif hp.order == IntegrationOrder.LINEAR:
      optimizer = TrapezoidalCollocationOptimizer(hp, cfg, system)
    else:  # Quadratic
      optimizer = HermiteSimpsonCollocationOptimizer(hp, cfg, system)
  elif hp.optimizer == OptimizerType.SHOOTING:
    optimizer = MultipleShootingOptimizer(hp, cfg, system)
  elif hp.optimizer == OptimizerType.FBSM:
    optimizer = FBSM(hp, cfg, system)
  else:
    raise KeyError
  return optimizer


class TrapezoidalCollocationOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    """
    An optimizer that uses direct trapezoidal collocation.
      For reference, see https://epubs.siam.org/doi/10.1137/16M1062569
    Args:
      hp: Hyperparameters
      cfg: Additional hyperparameters
      system: The system on which to perform the optimization
    """
    num_intervals = hp.intervals  # Segments
    h = system.T / num_intervals  # Segment length
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape
    # print("the control shape is", control_shape)

    u_guess = jnp.zeros((num_intervals + 1, control_shape))

    if system.x_T is not None:
      # We need to handle the cases where a terminal bound is specified only for some state variables, not all
      row_guesses = []
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=num_intervals + 1).reshape(-1, 1)
        else:
          _, row_guess = integrate(system.dynamics, system.x_0, u_guess, h, num_intervals, None, hp.order)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else:  # no final state requirement
      _, x_guess = integrate(system.dynamics, system.x_0, u_guess, h, num_intervals, None, hp.order)
    guess, unravel_decision_variables = ravel_pytree((x_guess, u_guess))
    self.x_guess, self.u_guess = x_guess, u_guess

    def trapezoid_cost(x_t1: jnp.ndarray, x_t2: jnp.ndarray, u_t1: float, u_t2: float, t1: float, t2: float) -> float:
      """
      Args:
        x_t1: State at start of interval
        x_t2: State at end of interval
        u_t1: Control at start of interval
        u_t2: Control at end of interval
        t1: Time at start of interval
        t2: Time at end of interval
      Returns:
        Trapezoid cost of the interval
      """
      return (h / 2) * (system.cost(x_t1, u_t1, t1) + system.cost(x_t2, u_t2, t2))

    def objective(variables: jnp.ndarray) -> float:
      """
      The objective function.
      Args:
        variables: Raveled state and decision variables
      Returns:
        The sum of the trapezoid costs across the whole trajectory
      """
      x, u = unravel_decision_variables(variables)
      t = jnp.linspace(0, system.T, num=num_intervals + 1)  # Support cost function with dependency on t
      cost = jnp.sum(vmap(trapezoid_cost)(x[:-1], x[1:], u[:-1], u[1:], t[:-1], t[1:]))
      if system.terminal_cost:
        cost += jnp.sum(system.terminal_cost_fn(x[-1], u[-1]))
      return cost

    def trapezoid_defect(x_t1: jnp.ndarray, x_t2: jnp.ndarray, u_t1: float, u_t2: float) -> jnp.ndarray:
      """
      Args:
        x_t1: State at start of interval
        x_t2: State at end of interval
        u_t1: Control at start of interval
        u_t2: Control at end of interval
      Returns:
        Trapezoid defect of the interval
      """
      left = (h / 2) * (system.dynamics(x_t1, u_t1) + system.dynamics(x_t2, u_t2))
      right = x_t2 - x_t1
      return left - right

    def constraints(variables: jnp.ndarray) -> jnp.ndarray:
      """
      The constraints function.
      Args:
        variables: Raveled state and decision variables
      Returns:
        An array of the defects of the whole trajectory
      """
      x, u = unravel_decision_variables(variables)
      return jnp.ravel(vmap(trapezoid_defect)(x[:-1], x[1:], u[:-1], u[1:]))

    ############################
    # State and Control Bounds #
    ############################
    x_bounds = np.empty((num_intervals + 1, system.bounds.shape[0] - control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]
    x_bounds[0, :, :] = np.expand_dims(system.x_0, 1)
    if system.x_T is not None:
      x_bounds[-control_shape, :, :] = np.expand_dims(system.x_T, 1)
    x_bounds = x_bounds.reshape((-1, 2))
    u_bounds = np.empty(((num_intervals + 1) * control_shape, 2))
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (num_intervals + 1):(control_shape - i + 1) * (num_intervals + 1)] = system.bounds[
        -i]
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(OptimizerType.COLLOCATION, hp, cfg, objective, constraints, bounds, guess,
                     unravel_decision_variables)


class HermiteSimpsonCollocationOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem):
    """
    An optimizer that uses direct Hermite-Simpson collocation.
      For reference, see https://epubs.siam.org/doi/10.1137/16M1062569
    Args:
      hp: Hyperparameters
      cfg: Additional hyperparameters
      system: The system on which to perform the optimization
    """
    num_intervals = hp.intervals
    interval_duration = system.T / hp.intervals
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    u_guess, mid_u_guess = jnp.zeros((num_intervals + 1, control_shape)), jnp.zeros((num_intervals, control_shape))

    # Initial guess for state and controls
    # O . . . O (guess dots)
    if system.x_T is not None:
      full_x_guess = jnp.linspace(system.x_0, system.x_T, num=hp.intervals + 1)

      # make the mid states linear too (interpolate between collocation points)
      mid_x_guess = np.copy((full_x_guess[:-1] + full_x_guess[1:]) / 2)
      # Now remove the end points from the state guesses
      x_guess = jnp.linspace(system.x_0, system.x_T, num=hp.intervals + 1)[1:-1]
    # O . . . . (guess dots)
    else:
      # _, x_guess = integrate(system.dynamics, system.x_0, u_guess, interval_duration, num_intervals)[1:] # should be right
      x_guess = jnp.ones(shape=(num_intervals, len(system.x_0))) * 0.25
      mid_x_guess = jnp.ones(shape=(num_intervals, len(system.x_0))) * 0.25

    # all_controls_guess, unravel_controls = ravel_pytree((u_guess, mid_u_guess))
    # all_states_guess, unravel_states = ravel_pytree((x_guess, mid_x_guess))

    initial_variables = (x_guess, mid_x_guess, u_guess, mid_u_guess)

    guess, unravel_decision_variables = ravel_pytree(initial_variables)
    self.x_guess, self.u_guess = x_guess, u_guess

    ############################
    # State and Control Bounds #
    ############################
    u_bounds = np.empty(((num_intervals + 1) * control_shape, 2))
    # for i in range(control_shape, 0, -1):
    #   u_bounds[(control_shape - i) * (num_intervals + 1):(control_shape - i + 1) * (num_intervals + 1)] = system.bounds[
    #     -i]

    u_bounds[:] = system.bounds[-1:]

    mid_u_bounds = np.empty((num_intervals * control_shape, 2))
    # for i in range(control_shape, 0, -1):
    #   u_bounds[(control_shape - i) * num_intervals:(control_shape - i + 1) * num_intervals] = system.bounds[-i]
    mid_u_bounds[:] = system.bounds[-1:]

    # print("u", self.u_guess.shape)
    # print("u bounds", u_bounds.shape)
    # u_bounds = np.empty(((num_intervals + 1) * control_shape, 2))
    # for i in range(control_shape, 0, -1):
    #   u_bounds[(control_shape - i) * (num_intervals + 1):(control_shape - i + 1) * (num_intervals + 1)] = system.bounds[
    #     -i]

    single_x_bounds = system.bounds[:-1].flatten()

    if system.x_T is not None:
      x_bounds = jnp.tile(single_x_bounds, reps=(num_intervals - 1, 1))
      mid_x_bounds = jnp.tile(single_x_bounds, reps=(num_intervals, 1))
    else:
      x_bounds = jnp.tile(single_x_bounds, reps=(num_intervals, 1))
      mid_x_bounds = jnp.tile(single_x_bounds, reps=(num_intervals, 1))

    x_bounds = x_bounds.reshape(-1, 2)
    mid_x_bounds = mid_x_bounds.reshape(-1, 2)

    bounds = jnp.vstack((x_bounds, mid_x_bounds, u_bounds, mid_u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    # For convenience
    def get_start_and_next_states_and_controls(variables: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                                                jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      """
      Extracts start, mid, and ending arrays of decision variables
      Args:
        variables: Raveled state and control variables
      Returns:
        (start xs, mid xs, end xs, start us, mid us, end us)
      """
      xs, mid_xs, us, mid_us = unravel_decision_variables(variables)

      if system.x_T is not None:
        starting_states = jnp.concatenate((system.x_0[jnp.newaxis], xs))
        desired_next_states = jnp.concatenate((xs, system.x_T[jnp.newaxis]))
      else:  # last "decision state" is actually final state in this case
        starting_states = jnp.concatenate((system.x_0[jnp.newaxis], xs[:-1]))
        desired_next_states = xs

      return starting_states, mid_xs, desired_next_states, \
             us[:-1], mid_us, us[1:]

    # Calculates midpoint constraints on-the-fly  TODO nh: is this comment necessary?
    def hs_defect(state, mid_state, next_state, control, mid_control, next_control):
      """
      Hermite-Simpson collocation constraints
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval
        next_control: Control at end of interval
      Returns:
        Hermite-Simpson defect of the interval
      """
      rhs = next_state - state
      lhs = (interval_duration / 6) \
            * (system.dynamics(state, control)
               + 4 * system.dynamics(mid_state, mid_control)
               + system.dynamics(next_state, next_control))
      return rhs - lhs

    def hs_interpolation(state, mid_state, next_state, control, mid_control, next_control):
      """
      Calculate Hermite-Simpson interpolation constraints
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval (unused)
        next_control: Control at end of interval
      Returns:
        Interpolation constraints
      """
      return (mid_state
              - (1 / 2) * (state + next_state)
              - (interval_duration / 8) * (system.dynamics(state, control) - system.dynamics(next_state, next_control)))

    # This is the "J" from the tutorial (6.5)
    def hs_cost(state, mid_state, next_state, control, mid_control, next_control):
      """
      Calculate the Hermite-Simpson cost.
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval
        next_control: Control at end of interval
      Returns:
        Hermite-Simpson cost of interval
      """
      return (interval_duration / 6) \
             * (system.cost(state, control)
                + 4 * system.cost(mid_state, mid_control)
                + system.cost(next_state, next_control))

    #######################
    # Cost and Constraint #
    #######################
    def objective(variables):
      """
      Calculate the Hermite-Simpson objective for this trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        Objective of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.sum(vmap(hs_cost)(*unraveled_vars))

    def hs_equality_constraints(variables):
      """
      Calculate the equality constraint violations for this trajectory (does not include midpoint constraints)
      Args:
        variables: Raveled states and controls
      Returns:
        Equality constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(hs_defect)(*unraveled_vars))

    def hs_interpolation_constraints(variables):
      """
      Calculate the midpoint constraint violations for this trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        Midpoint constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(hs_interpolation)(*unraveled_vars))

    def constraints(variables):
      """
      Calculate all constraint violations for this trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        All constraint violations of trajectory
      """
      equality_defects = hs_equality_constraints(variables)
      interpolation_defects = hs_interpolation_constraints(variables)
      return jnp.hstack((equality_defects, interpolation_defects))

    super().__init__(OptimizerType.COLLOCATION, hp, cfg, objective, constraints, bounds, guess,
                     unravel_decision_variables)


class MultipleShootingOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem, key: jax.random.PRNGKey = None):
    # TODO: make the key live in the hparams
    """
    An optimizer that uses performs direct multiple shooting.
      For reference, see https://epubs.siam.org/doi/book/10.1137/1.9780898718577
    Args:
      hp: Hyperparameters
      cfg: Additional hyperparameters
      system: The system on which to perform the optimization
    """
    num_steps = hp.intervals * hp.controls_per_interval
    step_size = system.T / num_steps
    interval_size = system.T / hp.intervals
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape
    midpoints_const = 2 if hp.order == IntegrationOrder.QUADRATIC else 1
    if key is None:
      self.key = jax.random.PRNGKey(hp.seed)
    else:
      self.key = key

    #################
    # Initial Guess #
    #################

    # Controls
    # TODO: decide if we like this way of guessing controls. If yes, then add it to the other optimizers too.
    self.key, subkey = jax.random.split(self.key)
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    # controls_guess = np.random.uniform(u_lower, u_upper,
    #                                    (midpoints_const * num_steps + 1, control_shape))

    if not (jnp.isfinite(u_lower) and jnp.isfinite(u_lower)):
      controls_guess = jnp.zeros_like((midpoints_const * num_steps + 1, control_shape))
    else:
      controls_guess = jax.random.uniform(subkey, (midpoints_const * num_steps + 1, control_shape),
                                          minval=u_lower, maxval=u_upper) * 0.01

    print("the controls guess is", controls_guess.shape)
    # controls_guess = jnp.zeros((midpoints_const * num_steps + 1, control_shape))

    # States
    if system.x_T is not None:
      row_guesses = []
      # For the state variables which have a required end state, interpolate between start and end;
      # otherwise, use rk4 with initial controls as a first guess at intermediate and end state values
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=hp.intervals + 1).reshape(-1, 1)
        else:
          _, row_guess = integrate(system.dynamics, system.x_0,
                                   controls_guess[::midpoints_const * hp.controls_per_interval], interval_size,
                                   hp.intervals, None, hp.order)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else:
      _, x_guess = integrate(system.dynamics, system.x_0, controls_guess[::midpoints_const * hp.controls_per_interval],
                             interval_size, hp.intervals, None, hp.order)
    guess, unravel = ravel_pytree((x_guess, controls_guess))
    assert len(x_guess) == hp.intervals + 1  # we have one state decision var for each node, including start and end
    self.x_guess, self.u_guess = x_guess, controls_guess

    # Augment the dynamics so we can integrate cost the same way we do state
    def augmented_dynamics(x_and_c: jnp.ndarray, u: float, t: float) -> jnp.ndarray:
      """
      Augments the dynamics with the cost function, so that all can be integrated together
      Args:
        x_and_c: State and current cost (current cost doesn't affect the cost calculation)
        u: Control
        t: Time
        custom_dynamics: Optional custom dynamics to replace system dynamics
      Returns:
        The cost of applying control u to state x at time t
      """
      x, c = x_and_c[:-1], x_and_c[-1]
      return jnp.append(system.dynamics(x, u), system.cost(x, u, t))

    def reorganize_controls(us):  # This still works, even for higher-order control shape
      """
      Reorganize controls into per-interval arrays
      Go from having controls like (num_controls + 1, control_shape) (left)
                           to like (hp.intervals, num_controls_per_interval + 1, control_shape) (right)
      [ 1. ,  1.1]                [ 1. ,  1.1]
      [ 2. ,  2.1]                [ 2. ,  2.1]
      [ 3. ,  3.1]                [ 3. ,  3.1]
      [ 4. ,  4.1]                [ 4. ,  4.1]
      [ 5. ,  5.1]
      [ 6. ,  6.1]                [ 4. ,  4.1]
      [ 7. ,  7.1]                [ 5. ,  5.1]
      [ 8. ,  8.1]                [ 6. ,  6.1]
      [ 9. ,  9.1]                [ 7. ,  7.1]
      [10. , 10.1]
                                  [ 7. ,  7.1]
                                  [ 8. ,  8.1]
                                  [ 9. ,  9.1]
                                  [10. , 10.1]
      Args:
        us: Controls
      Returns:
        Controls organized into per-interval arrays
      """
      new_controls = jnp.hstack(
        [us[:-1].reshape(hp.intervals, midpoints_const * hp.controls_per_interval, control_shape),
         us[::midpoints_const * hp.controls_per_interval][1:][:, jnp.newaxis]])
      # Needed for single shooting
      if len(new_controls.shape) == 3 and new_controls.shape[2] == 1:
        new_controls = new_controls.squeeze(axis=2)
      return new_controls

    def reorganize_times(ts):
      """
      Reorganize times into per-interval arrays
      Args:
        ts: Times
      Returns:
        Times organized into per-interval arrays
      """
      new_times = jnp.hstack([ts[:-1].reshape(hp.intervals, hp.controls_per_interval),
                              ts[::hp.controls_per_interval][1:][:, jnp.newaxis]])
      return new_times

    def objective(variables: jnp.ndarray) -> float:
      """
      Calculate the objective of a trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        The objective of the trajectory
      """
      # print("dynamics are", system.dynamics)
      # The commented code runs faster, but only does a linear interpolation for cost.
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
      reshaped_controls = reorganize_controls(us)

      t = jnp.linspace(0., system.T, num=num_steps + 1)
      t = reorganize_times(t)

      starting_xs_and_costs = jnp.hstack([xs[:-1], jnp.zeros(len(xs[:-1])).reshape(-1, 1)])

      # Integrate cost in parallel
      states_and_costs, _ = integrate_in_parallel(
        augmented_dynamics, starting_xs_and_costs, reshaped_controls,
        step_size, hp.controls_per_interval, t, hp.order)

      costs = jnp.sum(states_and_costs[:, -1])
      if system.terminal_cost:
        last_augmented_state = states_and_costs[-1]
        costs += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])

      return costs

    def constraints(variables: jnp.ndarray) -> jnp.ndarray:
      """
      Calculate the constraint violations of a trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        Constraint violations of trajectory
      """
      xs, us = unravel(variables)
      px, _ = integrate_time_independent_in_parallel(system.dynamics, xs[:-1], reorganize_controls(us), step_size,
                                                     hp.controls_per_interval, hp.order)
      return jnp.ravel(px - xs[1:])

    ############################
    # State and Control Bounds #
    ############################

    # State decision variables at every node
    x_bounds = np.zeros((hp.intervals + 1, system.bounds.shape[0] - control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]

    # Starting state
    x_bounds[0, :, :] = jnp.expand_dims(system.x_0, 1)

    # Ending state
    if system.x_T is not None:
      for i in range(len(system.x_T)):
        if system.x_T[i] is not None:
          x_bounds[-1, i, :] = system.x_T[i]

    # Reshape for ipopt's minimize
    x_bounds = x_bounds.reshape((-1, 2))

    # Control decision variables at every node, and if QUADRATIC order, also at midpoints
    u_bounds = np.empty(((midpoints_const * num_steps + 1) * control_shape, 2))  # Include midpoints too
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (midpoints_const * num_steps + 1):(control_shape - i + 1) * (
                midpoints_const * num_steps + 1)] = system.bounds[-i]

    # print("u bounds", u_bounds)
    # Stack all bounds together for the NLP solver
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(OptimizerType.SHOOTING, hp, cfg, objective, constraints, bounds, guess, unravel)


class FBSM(IndirectMethodOptimizer):  # Forward-Backward Sweep Method
  """
  The Forward-Backward Sweep Method, as described in Optimal Control Applied to Biological Models, Lenhart & Workman
  An iterative solver that, given an initial guess over the controls, will do a forward pass to retrieve the state
  variables trajectory followed by a backward pass to retrieve the adjoint variables trajectory. The optimality
  characterization is then used to update the control values.
  The process is repeated until convergence over the controls.
  """

  def __init__(self, hp: HParams, cfg: Config, system: IndirectFHCS):
    self.system = system
    self.N = hp.fbsm_intervals
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
      u_guess = jnp.zeros((self.N + 1, control_shape))
    if system.adj_T is not None:
      adj_guess = jnp.vstack((jnp.zeros((self.N, state_shape)), system.adj_T))
    else:
      adj_guess = jnp.zeros((self.N + 1, state_shape))
    self.t_interval = jnp.linspace(0, system.T, num=self.N + 1).reshape(-1, 1)

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
    """Helper function for `sequencesolver`
    """
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
    """Solve the continuous optimal problem with the Forward-Backward Sweep Method"""
    if self.terminal_cdtion:
      return self.sequencesolver()
    n = 0
    while n == 0 or self.stopping_criterion((self.x_guess, old_x), (self.u_guess, old_u), (self.adj_guess, old_adj)):
      old_u = self.u_guess.copy()
      old_x = self.x_guess.copy()
      old_adj = self.adj_guess.copy()

      self.x_guess = integrate_fbsm(self.system.dynamics, self.x_guess[0], self.u_guess, self.h, self.N,
                                    t=self.t_interval, discrete=self.system.discrete)[-1]
      self.adj_guess = integrate_fbsm(self.system.adj_ODE, self.adj_guess[-1], self.x_guess, -1 * self.h, self.N,
                                      self.u_guess, t=self.t_interval, discrete=self.system.discrete)[-1]

      u_estimate = self.system.optim_characterization(self.adj_guess, self.x_guess, self.t_interval)
      # Use basic convex approximation to update the guess on u
      self.u_guess = 0.5 * (u_estimate + old_u)

      n = n + 1

    return self.x_guess, self.u_guess, self.adj_guess

  def sequencesolver(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Implement the secant method for the special case where there is a terminal value on some state variables in
    addition to the initial values.
    """
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

      d = Va * (b - a) / (Vb - Va)
      b = a
      Vb = Va
      a = a - d
      self.reinitiate(a)
      x_a, _, _ = self.solve()
      Va = x_a[-1, self.term_cdtion_state] - self.term_value
      count += 1

    return self.x_guess, self.u_guess, self.adj_guess