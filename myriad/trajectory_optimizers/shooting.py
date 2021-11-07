# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp
import numpy as np

from jax.flatten_util import ravel_pytree

from myriad.config import Config, HParams, IntegrationMethod
from myriad.custom_types import Control, Params, Timestep
from myriad.systems import FiniteHorizonControlSystem
from myriad.utils import integrate_in_parallel, integrate_time_independent, integrate_time_independent_in_parallel
from myriad.trajectory_optimizers.base import TrajectoryOptimizer


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
    midpoints_const = 2 if hp.integration_method == IntegrationMethod.RK4 else 1
    if key is None:
      self.key = jax.random.PRNGKey(hp.seed)
    else:
      self.key = key

    #################
    # Initial Guess #
    #################

    # Controls
    # TODO: decide if we like this way of guessing controls. If yes, then add it to the other trajectory_optimizers too.
    self.key, subkey = jax.random.split(self.key)
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]

    controls_guess = jnp.zeros((midpoints_const * num_steps + 1, control_shape))

    # if jnp.isfinite(u_lower) and jnp.isfinite(u_upper):
    #   controls_guess += jax.random.normal(subkey, (midpoints_const * num_steps + 1, control_shape)) * (
    #             u_upper - u_lower) * 0.05

    print("the controls guess is", controls_guess.shape)

    # States
    if system.x_T is not None:
      row_guesses = []
      # For the state variables which have a required end state, interpolate between start and end;
      # otherwise, use rk4 with initial controls as a first guess at intermediate and end state values
      for i in range(0, len(system.x_T)):
        if system.x_T[i] is not None:
          row_guess = jnp.linspace(system.x_0[i], system.x_T[i], num=hp.intervals + 1).reshape(-1, 1)
        else:
          _, row_guess = integrate_time_independent(system.dynamics, system.x_0,
                                                    controls_guess[::midpoints_const * hp.controls_per_interval],
                                                    interval_size,
                                                    hp.intervals, hp.integration_method)
          row_guess = row_guess[:, i].reshape(-1, 1)
        row_guesses.append(row_guess)
      x_guess = jnp.hstack(row_guesses)
    else:
      _, x_guess = integrate_time_independent(system.dynamics, system.x_0,
                                              controls_guess[::midpoints_const * hp.controls_per_interval],
                                              interval_size, hp.intervals, hp.integration_method)
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

    # Augment the dynamics so we can integrate cost the same way we do state
    def parametrized_augmented_dynamics(params: Params, x_and_c: jnp.ndarray, u: Control, t: Timestep) -> jnp.ndarray:
      # TODO: docstring
      x, c = x_and_c[:-1], x_and_c[-1]
      return jnp.append(system.parametrized_dynamics(params, x, u), system.parametrized_cost(params, x, u, t))

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

    def parametrized_objective(params: Params, variables: jnp.ndarray) -> float:
      # TODO: docstring
      xs, us = unravel(variables)
      reshaped_controls = reorganize_controls(us)

      t = jnp.linspace(0., system.T, num=num_steps + 1)
      t = reorganize_times(t)

      starting_xs_and_costs = jnp.hstack([xs[:-1], jnp.zeros(len(xs[:-1])).reshape(-1, 1)])

      def dynamics(x_and_c: jnp.ndarray, u: Control, t: Timestep):
        return parametrized_augmented_dynamics(params, x_and_c, u, t)

      # Integrate cost in parallel
      states_and_costs, _ = integrate_in_parallel(
        dynamics, starting_xs_and_costs, reshaped_controls,
        step_size, hp.controls_per_interval, t, hp.integration_method)

      costs = jnp.sum(states_and_costs[:, -1])
      if system.terminal_cost:
        last_augmented_state = states_and_costs[-1]
        costs += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])

      return costs

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
        step_size, hp.controls_per_interval, t, hp.integration_method)

      costs = jnp.sum(states_and_costs[:, -1])
      if system.terminal_cost:
        last_augmented_state = states_and_costs[-1]
        costs += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])

      return costs

    def parametrized_constraints(params: Params, variables: jnp.ndarray) -> jnp.ndarray:
      """
      Calculate the constraint violations of a trajectory
      Args:
        variables: Raveled states and controls
        params: Dict of parameters for the model
      Returns:
        Constraint violations of trajectory
      """

      def dynamics(x_t: jnp.ndarray, u_t: jnp.ndarray):
        return system.parametrized_dynamics(params, x_t, u_t)

      xs, us = unravel(variables)
      px, _ = integrate_time_independent_in_parallel(dynamics, xs[:-1], reorganize_controls(us), step_size,
                                                     hp.controls_per_interval, hp.integration_method)
      return jnp.ravel(px - xs[1:])

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
                                                     hp.controls_per_interval, hp.integration_method)
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

    # Reshape for call to 'minimize'
    x_bounds = x_bounds.reshape((-1, 2))

    # Control decision variables at every node, and if RK4, also at midpoints
    u_bounds = np.empty(((midpoints_const * num_steps + 1) * control_shape, 2))  # Include midpoints too
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (midpoints_const * num_steps + 1):(control_shape - i + 1) * (
              midpoints_const * num_steps + 1)] = system.bounds[-i]

    # Reshape for call to 'minimize'
    u_bounds = u_bounds.reshape((-1, 2))

    # print("u bounds", u_bounds)
    # Stack all bounds together for the NLP solver
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    super().__init__(hp, cfg, objective, parametrized_objective, constraints, parametrized_constraints,
                     bounds, guess, unravel)
