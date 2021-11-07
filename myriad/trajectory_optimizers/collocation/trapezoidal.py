# (c) 2021 Nikolaus Howe
import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.flatten_util import ravel_pytree

from myriad.config import Config, HParams
from myriad.custom_types import Control, Cost, DState, Params, State, Timestep, DStates
from myriad.trajectory_optimizers.base import TrajectoryOptimizer
from myriad.systems import FiniteHorizonControlSystem
from myriad.utils import integrate_time_independent


class TrapezoidalCollocationOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem) -> None:
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

    ###########################
    # State and Control Guess #
    ###########################
    u_guess = jnp.zeros((num_intervals + 1, control_shape))

    if system.x_T is not None:
      # We need to handle the cases where a terminal bound is specified only for some state variables, not all
      row_guesses = []
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=num_intervals + 1).reshape(-1, 1)
        else:
          _, row_guess = integrate_time_independent(system.dynamics, system.x_0,
                                                    u_guess, h, num_intervals, hp.integration_method)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else:  # no final state requirement
      _, x_guess = integrate_time_independent(system.dynamics, system.x_0,
                                              u_guess, h, num_intervals, hp.integration_method)
    guess, unravel_decision_variables = ravel_pytree((x_guess, u_guess))
    self.x_guess, self.u_guess = x_guess, u_guess

    ############################
    # State and Control Bounds #
    ############################
    # Control bounds
    u_bounds = np.empty(((num_intervals + 1) * control_shape, 2))
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (num_intervals + 1)
               :(control_shape - i + 1) * (num_intervals + 1)] = system.bounds[-i]

    # Reshape to work with NLP solver
    u_bounds = u_bounds.reshape((-1, 2))

    # State bounds
    x_bounds = np.empty((num_intervals + 1, system.bounds.shape[0] - control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]
    x_bounds[0, :, :] = np.expand_dims(system.x_0, 1)
    if system.x_T is not None:
      x_bounds[-control_shape, :, :] = np.expand_dims(system.x_T, 1)

    # Reshape to work with NLP solver
    x_bounds = x_bounds.reshape((-1, 2))

    # Put control and state bounds together
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    def trapezoid_cost(x_t1: State, x_t2: State,
                       u_t1: Control, u_t2: Control,
                       t1: Timestep, t2: Timestep) -> Cost:
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

    def parametrized_trapezoid_cost(params: Params,
                                    x_t1: State, x_t2: State,
                                    u_t1: Control, u_t2: Control,
                                    t1: Timestep, t2: Timestep) -> Cost:
      """
      Args:
        x_t1: State at start of interval
        x_t2: State at end of interval
        u_t1: Control at start of interval
        u_t2: Control at end of interval
        t1: Time at start of interval
        t2: Time at end of interval
        params: Custom model parameters
      Returns:
        Trapezoid cost of the interval
      """
      return (h / 2) * (system.parametrized_cost(params, x_t1, u_t1, t1)
                        + system.parametrized_cost(params, x_t2, u_t2, t2))

    def objective(variables: jnp.ndarray) -> Cost:
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

    def parametrized_objective(params: Params, variables: jnp.ndarray) -> Cost:
      """
      The objective function.
      Args:
        variables: Raveled state and decision variables
        params: Custom model parameters
      Returns:
        The sum of the trapezoid costs across the whole trajectory
      """
      x, u = unravel_decision_variables(variables)
      t = jnp.linspace(0, system.T, num=num_intervals + 1)  # Support cost function with dependency on t
      cost = jnp.sum(vmap(parametrized_trapezoid_cost, in_axes=(None, 0, 0, 0, 0, 0, 0))(params,
                                                                                         x[:-1], x[1:],
                                                                                         u[:-1], u[1:],
                                                                                         t[:-1], t[1:]))
      if system.terminal_cost:
        cost += jnp.sum(system.terminal_cost_fn(x[-1], u[-1]))
      return cost
    # TODO: should the terminal cost function also take parameters?
    # probably yes... (will need to fix this in shooting and hs too then)

    def trapezoid_defect(x_t1: State, x_t2: State, u_t1: Control, u_t2: Control) -> DState:
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

    def parametrized_trapezoid_defect(params: Params,
                                      x_t1: State, x_t2: State,
                                      u_t1: Control, u_t2: Control) -> DState:
      """
      Args:
        x_t1: State at start of interval
        x_t2: State at end of interval
        u_t1: Control at start of interval
        u_t2: Control at end of interval
        params: Custom model parameters
      Returns:
        Trapezoid defect of the interval
      """
      left = (h / 2) * (system.parametrized_dynamics(params, x_t1, u_t1)
                        + system.parametrized_dynamics(params, x_t2, u_t2))
      right = x_t2 - x_t1
      return left - right

    def constraints(variables: jnp.ndarray) -> DStates:
      """
      The constraints function.
      Args:
        variables: Raveled state and decision variables
      Returns:
        An array of the defects of the whole trajectory
      """
      x, u = unravel_decision_variables(variables)
      return jnp.ravel(vmap(trapezoid_defect)(x[:-1], x[1:], u[:-1], u[1:]))

    def parametrized_constraints(params: Params, variables: jnp.ndarray) -> DStates:
      """
      The constraints function.
      Args:
        variables: Raveled state and decision variables
        params: Custom model parameters
      Returns:
        An array of the defects of the whole trajectory
      """
      x, u = unravel_decision_variables(variables)
      return jnp.ravel(vmap(parametrized_trapezoid_defect, in_axes=(None, 0, 0, 0, 0, 0, 0))(params,
                                                                                             x[:-1], x[1:],
                                                                                             u[:-1], u[1:]))

    super().__init__(hp, cfg, objective, parametrized_objective, constraints, parametrized_constraints,
                     bounds, guess, unravel_decision_variables)
