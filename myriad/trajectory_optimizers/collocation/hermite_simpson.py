# (c) 2021 Nikolaus Howe
import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.flatten_util import ravel_pytree
from typing import Tuple

from myriad.config import Config, HParams
from myriad.custom_types import Control, Controls, Cost, DState, DStates, Params, State, States, Timestep
from myriad.systems import FiniteHorizonControlSystem
from myriad.trajectory_optimizers.base import TrajectoryOptimizer


class HermiteSimpsonCollocationOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem) -> None:
    """
    An optimizer that uses direct Hermite-Simpson collocation.
      For reference, see https://epubs.siam.org/doi/10.1137/16M1062569.
      Note that we are keeping the knot points and the midpoints together
      in one big array, instead of separating them. This improves compatibility
      with the other trajectory_optimizers.
    Args:
      hp: Hyperparameters
      cfg: Additional hyperparameters
      system: The system on which to perform the optimization
    """
    interval_duration = system.T / hp.intervals
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    ###########################
    # State and Control Guess #
    ###########################

    # Initial guess for controls
    u_guess = jnp.zeros((2 * hp.intervals + 1, control_shape))

    # Initial guess for state
    if system.x_T is not None:
      x_guess = jnp.linspace(system.x_0, system.x_T, num=2 * hp.intervals + 1)
    else:
      x_guess = jnp.ones(shape=(2 * hp.intervals + 1, state_shape)) * 0.1

    initial_variables = (x_guess, u_guess)

    guess, unravel_decision_variables = ravel_pytree(initial_variables)
    self.x_guess, self.u_guess = x_guess, u_guess

    ############################
    # State and Control Bounds #
    ############################

    # Bounds for states
    x_bounds = np.zeros((2 * hp.intervals + 1, system.bounds.shape[0] - control_shape, 2))
    x_bounds[:, :, :] = system.bounds[:-control_shape]

    # Starting state
    x_bounds[0, :, :] = jnp.expand_dims(system.x_0, 1)

    # Ending state
    if system.x_T is not None:
      for i in range(len(system.x_T)):
        if system.x_T[i] is not None:
          x_bounds[-1, i, :] = system.x_T[i]

    # Reshape for call to 'minimize'
    x_bounds = x_bounds.reshape((-1, 2))

    # Bounds for controls
    u_bounds = np.empty(((2 * hp.intervals + 1) * control_shape, 2))  # Include midpoints too
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (2 * hp.intervals + 1):(control_shape - i + 1) * (
              2 * hp.intervals + 1)] = system.bounds[-i]

    # Reshape for call to 'minimize'
    u_bounds = u_bounds.reshape((-1, 2))

    # Stack all bounds together for the NLP solver
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    # Helper function
    def get_start_and_next_states_and_controls(variables: jnp.ndarray) -> Tuple[States, States, States,
                                                                                Controls, Controls, Controls]:
      """
      Extracts start, mid, and ending arrays of decision variables
      Args:
        variables: Raveled state and control variables
      Returns:
        (start xs, mid xs, end xs, start us, mid us, end us)
      """
      xs, us = unravel_decision_variables(variables)

      # States
      knot_point_xs = xs[::2]
      start_xs = knot_point_xs[:-1]
      end_xs = knot_point_xs[1:]
      mid_point_xs = xs[1::2]

      # Controls
      knot_point_us = us[::2]
      start_us = knot_point_us[:-1]
      end_us = knot_point_us[1:]
      mid_point_us = us[1::2]

      return start_xs, mid_point_xs, end_xs, start_us, mid_point_us, end_us

    # Calculates midpoint constraint on-the-fly
    def hs_defect(state: State, mid_state: State, next_state: State,
                  control: Control, mid_control: Control, next_control: Control) -> DState:
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
      lhs = (interval_duration / 6) * (system.dynamics(state, control)
                                       + 4 * system.dynamics(mid_state, mid_control)
                                       + system.dynamics(next_state, next_control))
      return rhs - lhs

    # Calculates midpoint constraint on-the-fly
    def parametrized_hs_defect(params: Params,
                               state: State, mid_state: State, next_state: State,
                               control: Control, mid_control: Control, next_control: Control) -> DState:
      """
      Hermite-Simpson collocation constraints
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval
        next_control: Control at end of interval
        params: Custom model parameters
      Returns:
        Hermite-Simpson defect of the interval
      """
      rhs = next_state - state
      lhs = (interval_duration / 6) * (system.parametrized_dynamics(params, state, control)
                                       + 4 * system.parametrized_dynamics(params, mid_state, mid_control)
                                       + system.parametrized_dynamics(params, next_state, next_control))
      return rhs - lhs

    def hs_interpolation(state: State, mid_state: State, next_state: State,
                         control: Control, mid_control: Control, next_control: Control) -> DState:
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
        Interpolation constraint
      """
      return (mid_state
              - (1 / 2) * (state + next_state)
              - (interval_duration / 8) * (system.dynamics(state, control)
                                           - system.dynamics(next_state, next_control)))

    def parametrized_hs_interpolation(params: Params,
                                      state: State, mid_state: State, next_state: State,
                                      control: Control, mid_control: Control, next_control: Control) -> DState:
      """
      Calculate Hermite-Simpson interpolation constraints
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval (unused)
        next_control: Control at end of interval
        params: Custom model parameters
      Returns:
        Interpolation constraint
      """
      return (mid_state
              - (1 / 2) * (state + next_state)
              - (interval_duration / 8) * (system.parametrized_dynamics(params, state, control)
                                           - system.parametrized_dynamics(params, next_state, next_control)))

    # This is the "J" from the tutorial (6.5)
    def hs_cost(state: State, mid_state: State, next_state: State,
                control: Control, mid_control: Control, next_control: Control,
                start_time: Timestep, mid_time: Timestep, next_time: Timestep) -> Cost:
      """
      Calculate the Hermite-Simpson cost.
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval
        next_control: Control at end of interval
        start_time: Time at start of interval
        mid_time: Time at midpoint of interval
        next_time: Time at end of interval
      Returns:
        Hermite-Simpson cost of interval
      """
      return (interval_duration / 6) * (system.cost(state, control, start_time)
                                        + 4 * system.cost(mid_state, mid_control, mid_time)
                                        + system.cost(next_state, next_control, next_time))

    def parametrized_hs_cost(params: Params,
                             state: State, mid_state: State, next_state: State,
                             control: Control, mid_control: Control, next_control: Control,
                             start_time: Timestep, mid_time: Timestep, next_time: Timestep) -> Cost:
      """
      Calculate the Hermite-Simpson cost.
      Args:
        state: State at start of interval
        mid_state: State at midpoint of interval
        next_state: State at end of interval
        control: Control at start of interval
        mid_control: Control at midpoint of interval
        next_control: Control at end of interval
        start_time: Time at start of interval
        mid_time: Time at midpoint of interval
        next_time: Time at end of interval
        params: Custom model parameters
      Returns:
        Hermite-Simpson cost of interval
      """
      return (interval_duration / 6) * (system.parametrized_cost(params, state, control, start_time)
                                        + 4 * system.parametrized_cost(params, mid_state, mid_control, mid_time)
                                        + system.parametrized_cost(params, next_state, next_control, next_time))

    #######################
    # Cost and Constraint #
    #######################
    def objective(variables: jnp.ndarray) -> Cost:
      """
      Calculate the Hermite-Simpson objective for this trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        Objective of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      all_times = jnp.linspace(0, system.T, num=2 * hp.intervals + 1)  # Support cost function with dependency on t
      start_and_end_times = all_times[::2]
      start_times = start_and_end_times[:-1]
      end_times = start_and_end_times[1:]
      mid_times = all_times[1::2]
      return jnp.sum(vmap(hs_cost)(*unraveled_vars, start_times, mid_times, end_times))

    def parametrized_objective(params: Params, variables: jnp.ndarray) -> Cost:
      """
      Calculate the Hermite-Simpson objective for this trajectory
      Args:
        variables: Raveled states and controls
        params: Custom model parameters
      Returns:
        Objective of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      all_times = jnp.linspace(0, system.T, num=2 * hp.intervals + 1)  # Support cost function with dependency on t
      start_and_end_times = all_times[::2]
      start_times = start_and_end_times[:-1]
      end_times = start_and_end_times[1:]
      mid_times = all_times[1::2]
      return jnp.sum(vmap(parametrized_hs_cost, in_axes=(None, 0, 0, 0, 0, 0, 0))(params,
                                                                                  *unraveled_vars, start_times,
                                                                                  mid_times, end_times))
    # TODO: test to make sure this actually works ^

    def hs_equality_constraints(variables: jnp.ndarray) -> DStates:
      """
      Calculate the equality constraint violations for this trajectory (does not include midpoint constraints)
      Args:
        variables: Raveled states and controls
      Returns:
        Equality constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(hs_defect)(*unraveled_vars))

    def parametrized_hs_equality_constraints(params: Params, variables: jnp.ndarray) -> DStates:
      """
      Calculate the equality constraint violations for this trajectory (does not include midpoint constraints)
      Args:
        variables: Raveled states and controls
        params: Custom model parameters
      Returns:
        Equality constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(parametrized_hs_defect, in_axes=(None, 0, 0, 0, 0, 0, 0))(params, *unraveled_vars))

    def hs_interpolation_constraints(variables: jnp.ndarray) -> DStates:
      """
      Calculate the midpoint constraint violations for this trajectory
      Args:
        variables: Raveled states and controls
      Returns:
        Midpoint constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(hs_interpolation)(*unraveled_vars))

    def parametrized_hs_interpolation_constraints(params: Params, variables: jnp.ndarray) -> DStates:
      """
      Calculate the midpoint constraint violations for this trajectory
      Args:
        variables: Raveled states and controls
        params: Custom model parameters
      Returns:
        Midpoint constraint violations of trajectory
      """
      unraveled_vars = get_start_and_next_states_and_controls(variables)
      return jnp.ravel(vmap(parametrized_hs_interpolation, in_axes=(None, 0, 0, 0, 0, 0, 0))(params, *unraveled_vars))

    def constraints(variables: jnp.ndarray) -> DStates:
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

    def parametrized_constraints(params: Params, variables: jnp.ndarray) -> DStates:
      """
      Calculate all constraint violations for this trajectory
      Args:
        variables: Raveled states and controls
        params: Custom model parameters
      Returns:
        All constraint violations of trajectory
      """
      equality_defects = parametrized_hs_equality_constraints(params, variables)
      interpolation_defects = parametrized_hs_interpolation_constraints(params, variables)
      return jnp.hstack((equality_defects, interpolation_defects))

    super().__init__(hp, cfg, objective, parametrized_objective, constraints, parametrized_constraints,
                     bounds, guess, unravel_decision_variables)
