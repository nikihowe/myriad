# (c) 2021 Nikolaus Howe
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax.flatten_util import ravel_pytree

# from scipy.optimize import minimize
from jax.scipy.optimize import minimize

from myriad.config import Config, HParams, IntegrationMethod
from myriad.custom_types import Control, Params, Timestep, Controls
from myriad.systems import FiniteHorizonControlSystem
from myriad.utils import integrate_in_parallel, integrate_time_independent, integrate_time_independent_in_parallel, \
  integrate
from myriad.trajectory_optimizers.base import TrajectoryOptimizer


class UnconstrainedShootingOptimizer(TrajectoryOptimizer):
  def __init__(self, hp: HParams, cfg: Config, system: FiniteHorizonControlSystem, key: jax.random.PRNGKey = None):
    num_steps = hp.intervals * hp.controls_per_interval
    step_size = system.T / num_steps
    interval_size = system.T / hp.intervals
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape
    midpoints_const = 1
    if key is None:
      self.key = jax.random.PRNGKey(hp.seed)
    else:
      self.key = key

    #################
    # Initial Guess #
    #################

    # Controls
    self.key, subkey = jax.random.split(self.key)
    controls_guess = jnp.zeros((midpoints_const * num_steps + 1, control_shape))
    print("the controls guess is", controls_guess.shape)

    guess = controls_guess
    self.u_guess = guess

    # Augment the dynamics so we can integrate cost the same way we do state
    def augmented_dynamics(x_and_c: jnp.ndarray, u: float, t: float) -> jnp.ndarray:
      x, c = x_and_c[:-1], x_and_c[-1]
      return jnp.append(system.dynamics(x, u), system.cost(x, u, t))

    def objective(controls: Controls) -> float:
      us = controls
      reshaped_controls = us.reshape(101, 1)

      t = jnp.linspace(0., system.T, num=num_steps + 1)

      starting_xs_and_costs = jnp.hstack([system.x_0, 0.])  # .reshape(2, 1)

      states_and_costs, _ = integrate(
        augmented_dynamics, starting_xs_and_costs, reshaped_controls,
        step_size, hp.controls_per_interval, t, hp.integration_method)

      costs = states_and_costs[-1]
      if system.terminal_cost:
        last_augmented_state = states_and_costs[-1]
        costs += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])

      return costs

    def constraints(controls: Controls) -> jnp.ndarray:
      return jnp.zeros((1,))

    ############################
    # State and Control Bounds #
    ############################

    # Control decision variables at every node, and if RK4, also at midpoints
    u_bounds = np.empty(((midpoints_const * num_steps + 1) * control_shape, 2))  # Include midpoints too
    for i in range(control_shape, 0, -1):
      u_bounds[(control_shape - i) * (midpoints_const * num_steps + 1):(control_shape - i + 1) * (
              midpoints_const * num_steps + 1)] = system.bounds[-i]

    # Reshape for call to 'minimize'
    u_bounds = u_bounds.reshape((-1, 2))

    # print("u bounds", u_bounds)
    # Stack all bounds together for the NLP solver
    bounds = u_bounds
    self.u_bounds = u_bounds

    def parametrized_objective(*args):
      pass

    def parametrized_constraints(*args):
      pass

    def unravel(_):
      pass

    super().__init__(hp, cfg, objective, parametrized_objective, constraints, parametrized_constraints,
                     bounds, guess, unravel)

  def unconstrained_solve(self):
    print("going to solve")
    # print("controls", self.guess.shape)
    # print("bounds", self.bounds.shape)
    # opt_inputs = {
    #   'method': "BFGS",
    #   'fun': jax.jit(self.objective),
    #   'x0': self.guess.squeeze(),
    #   # 'bounds': self.bounds,
    #   # 'jac': jax.jit(jax.grad(self.objective)),
    #   'options': {"maxiter": 1000}
    # }

    # print("before minimizing, let's calculate the objective")
    # self.objective(self.guess)
    # raise SystemExit
    # start = time.time()
    # res = minimize(**opt_inputs)
    # print("took", time.time() - start)

    start = time.time()
    res2 = self.constrained_gradient_descent()
    print("took", time.time() - start)

    return res2

  def constrained_gradient_descent(self):
    learning_rate = 0.01
    max_iter = 5_000
    jitted_objective = jax.jit(self.objective)
    guess = jnp.array(self.guess)

    @jax.jit
    def update(guess):
      return jnp.clip(guess - learning_rate * jax.grad(jitted_objective)(guess),
                      a_min=self.bounds[:, :1], a_max=self.bounds[:, 1:])

    for i in range(max_iter):
      guess = update(guess)

    class Res:
      def __init__(self):
        self.x = guess

    return Res()
