# (c) 2021 Nikolaus Howe
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import time
import typing

if typing.TYPE_CHECKING:
  from myriad.neural_ode.create_node import NeuralODE
  from myriad.config import HParams, Config

from jax import jit, lax, vmap
from typing import Callable, Optional, Tuple, Dict

from myriad.config import Config, HParams, IntegrationMethod, SamplingApproach
from myriad.systems import FiniteHorizonControlSystem
from myriad.custom_types import Control, Controls, Dataset, DState, State, States, Cost, Timestep


def integrate(
        dynamics_t: Callable[[State, Control, Timestep], DState],  # dynamics function
        x_0: State,  # starting state
        interval_us: Controls,  # controls
        h: float,  # step size
        N: int,  # steps
        ts: jnp.ndarray,  # times
        integration_method: IntegrationMethod  # allows user to choose interpolation for controls
) -> Tuple[State, States]:
  # QUESTION: do we want to keep this interpolation for rk4, or move to linear?
  @jit
  def rk4_step(x, u1, u2, u3, t):
    k1 = dynamics_t(x, u1, t)
    k2 = dynamics_t(x + h * k1 / 2, u2, t + h / 2)
    k3 = dynamics_t(x + h * k2 / 2, u2, t + h / 2)
    k4 = dynamics_t(x + h * k3, u3, t + h)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @jit
  def heun_step(x, u1, u2, t):
    k1 = dynamics_t(x, u1, t)
    k2 = dynamics_t(x + h * k1, u2, t + h)
    return x + h / 2 * (k1 + k2)

  @jit
  def midpoint_step(x, u1, u2, t):
    x_mid = x + h * dynamics_t(x, u1, t)
    u_mid = (u1 + u2) / 2
    return x + h * dynamics_t(x_mid, u_mid, t + h / 2)

  @jit
  def euler_step(x, u, t):
    return x + h * dynamics_t(x, u, t)

  def fn(carried_state, idx):
    if integration_method == IntegrationMethod.EULER:
      one_step_forward = euler_step(carried_state, interval_us[idx], ts[idx])
    elif integration_method == IntegrationMethod.HEUN:
      one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx + 1], ts[idx])
    elif integration_method == IntegrationMethod.MIDPOINT:
      one_step_forward = midpoint_step(carried_state, interval_us[idx], interval_us[idx + 1], ts[idx])
    elif integration_method == IntegrationMethod.RK4:
      one_step_forward = rk4_step(carried_state, interval_us[2 * idx], interval_us[2 * idx + 1],
                                  interval_us[2 * idx + 2], ts[idx])
    else:
      print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
      raise KeyError

    return one_step_forward, one_step_forward  # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


# Used for the augmented state cost calculation
integrate_in_parallel = vmap(integrate, in_axes=(None, 0, 0, None, None, 0, None))  # , static_argnums=(0, 5, 6)


def integrate_time_independent(
        dynamics_t: Callable[[State, Control], DState],  # dynamics function
        x_0: State,  # starting state
        interval_us: Controls,  # controls
        h: float,  # step size
        N: int,  # steps
        integration_method: IntegrationMethod  # allows user to choose int method
) -> Tuple[State, States]:
  # QUESTION: do we want to keep the mid-controls as decision variables for RK4,
  # or move to simply taking the average between the edge ones?
  @jit
  def rk4_step(x, u1, u2, u3):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h * k1 / 2, u2)
    k3 = dynamics_t(x + h * k2 / 2, u2)
    k4 = dynamics_t(x + h * k3, u3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @jit
  def heun_step(x, u1, u2):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h * k1, u2)
    return x + h / 2 * (k1 + k2)

  @jit
  def midpoint_step(x, u1, u2):
    x_mid = x + h * dynamics_t(x, u1)
    u_mid = (u1 + u2) / 2
    return x + h * dynamics_t(x_mid, u_mid)

  @jit
  def euler_step(x, u):
    return x + h * dynamics_t(x, u)

  def fn(carried_state, idx):
    if integration_method == IntegrationMethod.EULER:
      one_step_forward = euler_step(carried_state, interval_us[idx])
    elif integration_method == IntegrationMethod.HEUN:
      one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx + 1])
    elif integration_method == IntegrationMethod.MIDPOINT:
      one_step_forward = midpoint_step(carried_state, interval_us[idx], interval_us[idx + 1])
    elif integration_method == IntegrationMethod.RK4:
      one_step_forward = rk4_step(carried_state, interval_us[2 * idx], interval_us[2 * idx + 1],
                                  interval_us[2 * idx + 2])
    else:
      print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
      raise KeyError

    return one_step_forward, one_step_forward  # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


integrate_time_independent_in_parallel = vmap(integrate_time_independent, in_axes=(None, 0, 0, None, None, None))


# Used for the adjoint integration
def integrate_fbsm(
        dynamics_t: Callable[[State, Control, Optional[jnp.ndarray], Optional[jnp.ndarray]],
                             jnp.ndarray],  # dynamics function
        x_0: jnp.ndarray,  # starting state
        u: jnp.ndarray,  # controls
        h: float,  # step size  # is negative in backward mode
        N: int,  # steps
        v: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        discrete: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Implementation of Runge-Kutta 4th order method for ODE solving, adapted for the FBSM method.
  Specifically, it can either performs a numerical integration in a forward sweep over the states variables,
  or a backward sweep to integrate over the adjoint variables.
  Args:
    dynamics_t: (Callable) -- The dynamics (ODEs) to integrate
    x_0: The initial value to begin integration
    u: (jnp.ndarray) -- A guess over a costate variable.
    h: (float) -- The step size for the numerical integration
    N: (int) -- The number of steps for the numerical integration
    v: (jnp.ndarray, optional) -- Another costate variable, if needed
    t: (jnp.ndarray, optional) -- The time variable, for time-dependent dynamics
    discrete: (bool, optional) -- Perform direct calculation instead of integration if facing a discrete system.
  Returns:
    final_state, trajectory : Tuple[jnp.ndarray, jnp.array] -- The final value of the integrated variable and the complete trajectory
  """

  @jit
  def rk4_step(x_t1, u, u_next, v, v_next, t):
    u_convex_approx = (u + u_next) / 2
    v_convex_approx = (v + v_next) / 2

    k1 = dynamics_t(x_t1, u, v, t)
    k2 = dynamics_t(x_t1 + h * k1 / 2, u_convex_approx, v_convex_approx, t + h / 2)
    k3 = dynamics_t(x_t1 + h * k2 / 2, u_convex_approx, v_convex_approx, t + h / 2)
    k4 = dynamics_t(x_t1 + h * k3, u_next, v_next, t + h)

    return x_t1 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

  if v is None:
    v = jnp.empty_like(u)
  if t is None:
    t = jnp.empty_like(u)

  direction = int(jnp.sign(h))
  if discrete:
    if direction >= 0:
      fn = lambda x_t, idx: [dynamics_t(x_t, u[idx], v[idx], t[idx])] * 2
    else:
      fn = lambda x_t, idx: [dynamics_t(x_t, u[idx], v[idx - 1], t[idx - 1])] * 2
  else:
    fn = lambda x_t, idx: [rk4_step(x_t, u[idx], u[idx + direction], v[idx], v[idx + direction], t[idx])] * 2
  if direction >= 0:
    x_T, ys = lax.scan(fn, x_0, jnp.arange(N))
    return x_T, jnp.concatenate((x_0[None], ys))

  else:
    x_T, ys = lax.scan(fn, x_0, jnp.arange(N, 0, -1))
    return x_T, jnp.concatenate((jnp.flipud(ys), x_0[None]))


# First, get the optimal controls and resulting trajectory using the true system model.
# Then, replace the model dynamics with the trained neural network,
# and use that to find the "optimal" controls according to the NODE model.
# Finally get the resulting true state trajectory coming from those suboptimal controls.
# def plan_with_model(node: NeuralODE, regularize: bool = False) -> Controls:
#   apply_net = lambda x, u: node.net.apply(node.params, jnp.append(x, u))  # use nonlocal net and params
#
#   # Replace system dynamics, but remember it to restore later
#   # old_dynamics = node.system.dynamics
#   # node.system.dynamics = apply_net
#
#   objective = functools.partial(node.optimizer.objective, custom_dynamics=apply_net)
#   constraints = functools.partial(node.optimizer.constraints, custom_dynamics=apply_net)
#
#   opt_inputs = {
#     'objective': objective,
#     'guess': node.optimizer.guess,
#     'constraints': constraints,
#     'bounds': node.optimizer.bounds,
#     'unravel': node.optimizer.unravel
#   }
#
#   _, u = solve(node.hp, node.cfg, opt_inputs)
#
#   # Restore system dynamics
#   # node.system.dynamics = old_dynamics
#
#   return u.squeeze()  # this is necessary for later broadcasting


def plan_with_node_model(node: NeuralODE) -> Tuple[States, Controls]:
  apply_net = lambda x, u: node.net.apply(node.params, jnp.append(x, u))  # use nonlocal net and params

  # Replace system dynamics, but remember it to restore later
  old_dynamics = node.system.dynamics
  node.system.dynamics = apply_net

  solved_results = node.optimizer.solve()

  # Restore system dynamics
  node.system.dynamics = old_dynamics

  return solved_results['x'], solved_results['u']
  # TODO: I'm removing the squeeze on the ['u'] because it's causing problems later on
  # hopefully this doesn't break something else... all in the name of supporting vector controls


# Find the optimal trajectory according the learned model
def get_optimal_node_trajectory(node: NeuralODE) -> Tuple[States, Controls]:
  _, opt_u = plan_with_node_model(node)
  _, opt_x = integrate_time_independent(node.system.dynamics, node.system.x_0, opt_u,
                                        node.stepsize, node.num_steps, node.hp.integration_method)
  # assert not jnp.isnan(opt_u).all() and not jnp.isnan(opt_x).all()
  # NOTE: it used to return things in the opposite order! might cause bugs!
  return opt_x, opt_u


# TODO: make the start state default to the system start state
def get_state_trajectory_and_cost(hp: HParams, system: FiniteHorizonControlSystem,
                                  start_state: State, us: Controls) -> Tuple[States, Cost]:
  @jax.jit
  def augmented_dynamics(x_and_c: jnp.ndarray, u: Control, t: Timestep) -> jnp.ndarray:
    x, c = x_and_c[:-1], x_and_c[-1]
    return jnp.append(system.dynamics(x, u), system.cost(x, u, t))

  num_steps = hp.intervals * hp.controls_per_interval
  step_size = system.T / num_steps
  times = jnp.linspace(0., system.T, num=num_steps + 1)
  starting_x_and_cost = jnp.append(start_state, 0.)

  # print("starting x and cost", starting_x_and_cost)
  # print("us", us.shape)
  # print("step size", step_size)
  # print("num steps", num_steps)
  # print("times", times.shape)
  # raise SystemExit
  # Integrate cost in parallel
  # print("entering integration")
  # print("the us are", us)
  _, state_and_cost = integrate(
    augmented_dynamics, starting_x_and_cost, us,
    step_size, num_steps, times, hp.integration_method)

  # print("the states and costs are", state_and_cost)
  # print("the states and costs are", state_and_cost.shape)
  # raise SystemExit

  states = state_and_cost[:, :-1]

  # print("extracted states", states.shape)

  last_augmented_state = state_and_cost[-1]
  # print("last aug state", last_augmented_state)
  cost = last_augmented_state[-1]
  # print("cost", cost)
  if system.terminal_cost:
    cost += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])
  # raise SystemExit
  return states, cost


def smooth(curve: jnp.ndarray, its: int) -> jnp.ndarray:
  curve = np.array(curve)
  kernel = np.array([0.15286624, 0.22292994, 0.24840764, 0.22292994, 0.15286624])  # Gaussian blur
  for it in range(its):
    for i, row in enumerate(curve):
      for j, dim in enumerate(row.T):
        dim = np.pad(dim, (2, 2), 'edge')
        dim = np.convolve(dim, kernel, mode='valid')
        curve[i, :, j] = dim
  return jnp.array(curve)


def get_defect(system: FiniteHorizonControlSystem, learned_xs: States) -> Optional[jnp.array]:
  defect = None
  if system.x_T is not None:
    defect = []
    for i, s in enumerate(learned_xs[-1]):
      if system.x_T[i] is not None:
        defect.append(s - system.x_T[i])

  if defect is not None:
    defect = jnp.array(defect)

  return defect


def generate_dataset(hp: HParams, cfg: Config,
                     given_us: Optional[Controls] = None) -> Dataset:
  system = hp.system()
  hp.key, subkey = jax.random.split(hp.key)

  # Generate |total dataset size| control trajectories
  total_size = hp.train_size + hp.val_size + hp.test_size

  # TODO: fix what happens in case of infinite bounds
  u_lower = system.bounds[hp.state_size:, 0]
  u_upper = system.bounds[hp.state_size:, 1]
  x_lower = system.bounds[:hp.state_size, 0]
  x_upper = system.bounds[:hp.state_size, 1]
  if jnp.isinf(u_lower).any() or jnp.isinf(u_upper).any():
    raise Exception("infinite control bounds, aborting")
  if jnp.isinf(x_lower).any() or jnp.isinf(x_upper).any():
    raise Exception("infinite state bounds, aborting")

  spread = (u_upper - u_lower) * hp.sample_spread

  ########################
  # RANDOM WALK CONTROLS #
  ########################
  if hp.sampling_approach == SamplingApproach.RANDOM_WALK:
    # Make all the first states
    all_start_us = np.random.uniform(u_lower, u_upper, (total_size, 1, hp.control_size))
    all_us = all_start_us

    for i in range(hp.num_steps):
      next_us = np.random.normal(0, spread, (total_size, 1, hp.control_size))
      rightmost_us = all_us[:, -1:, :]
      together = np.clip(next_us + rightmost_us, u_lower, u_upper)
      all_us = np.concatenate((all_us, together), axis=1)

  # elif hp.sampling_approach == SamplingApproach.RANDOM_GRID:
  #   single_ascending_controls = np.linspace(u_lower, u_upper, hp.num_steps + 1)
  #   parallel_ascending_controls = single_ascending_controls[np.newaxis].repeat(total_size)
  #   assert parallel_ascending_controls.shape == ()
  # NOTE: we could also generate data by exhaustively considering every combination
  #       of state-control pair up to some discretization. This might just solve
  #       the problem. Unfortunately, curse of dimensionality is real.
  # IDEA: let's try doing this on the CANCERTREATMENT domain, and see whether
  #       this is enough to help neural ODE figure out what is going on
  #       at the very start of planning

  ###########################
  # UNIFORM RANDOM CONTROLS #
  ###########################
  elif hp.sampling_approach == SamplingApproach.UNIFORM:
    all_us = jax.random.uniform(subkey, (total_size, hp.num_steps + 1, hp.control_size),
                                minval=u_lower, maxval=u_upper) * 0.75  # TODO
  # TODO: make sure having added control size everywhere didn't break things
  #########################
  # AROUND GIVEN CONTROLS #
  #########################
  elif hp.sampling_approach == SamplingApproach.TRUE_OPTIMAL or hp.sampling_approach == SamplingApproach.CURRENT_OPTIMAL:
    if given_us is None:
      print("Since you didn't provide any controls, we'll use a uniform random guess")
      all_us = jax.random.uniform(subkey, (total_size, hp.num_steps + 1, hp.control_size),
                                  minval=u_lower, maxval=u_upper) * 0.75  # TODO
      # raise Exception("If sampling around a control trajectory, need to provide that trajectory.")

    else:
      noise = jax.random.normal(key=subkey, shape=(total_size, hp.num_steps + 1, hp.control_size)) \
              * (u_upper - u_lower) * hp.sample_spread
      all_us = jnp.clip(given_us[jnp.newaxis].repeat(total_size, axis=0).squeeze() + noise.squeeze(), a_min=u_lower,
                        a_max=u_upper)

  else:
    raise Exception("Unknown sampling approach, please choose among", SamplingApproach.__dict__['_member_names_'])

  print("initial controls shape", all_us.shape)

  # Smooth the controls if so desired
  if hp.to_smooth:
    start = time.time()
    all_us = smooth(all_us, 2)
    end = time.time()
    print(f"smoothing took {end - start}s")

  # TODO: I really dislike having to have this line below. Is there no way to remove it?
  # Make the controls guess smaller so our dynamics don't explode
  # all_us *= 0.1

  # Generate the start states
  start_states = system.x_0[jnp.newaxis].repeat(total_size, axis=0)

  # Generate the states from applying the chosen controls
  if hp.start_spread > 0.:
    hp.key, subkey = jax.random.split(hp.key)
    start_states += jax.random.normal(subkey,
                                      shape=start_states.shape) * hp.start_spread  # TODO: explore different spreads
    start_states = jnp.clip(start_states, a_min=x_lower, a_max=x_upper)

  # Generate the corresponding state trajectories
  _, all_xs = integrate_time_independent_in_parallel(system.dynamics, start_states,
                                                     all_us, hp.stepsize, hp.num_steps,
                                                     hp.integration_method)

  # Noise up the state observations
  hp.key, subkey = jax.random.split(hp.key)
  all_xs += jax.random.normal(subkey, shape=all_xs.shape) * (x_upper - x_lower) * hp.noise_level
  all_xs = jnp.clip(all_xs, a_min=x_lower, a_max=x_upper)

  # Stack the states and controls together
  xs_and_us = jnp.concatenate((all_xs, all_us), axis=2)

  if cfg.verbose:
    print("Generating training control trajectories between bounds:")
    print("  u lower", u_lower)
    print("  u upper", u_upper)
    print("of shapes:")
    print("  xs shape", all_xs.shape)
    print("  us shape", all_us.shape)
    print("  together", xs_and_us.shape)

  assert np.isfinite(xs_and_us).all()
  return xs_and_us


def yield_minibatches(hp: HParams, total_size: int, dataset: Dataset) -> iter:
  assert total_size <= dataset.shape[0]

  tmp_dataset = np.random.permutation(dataset)
  num_minibatches = total_size // hp.minibatch_size + (1 if total_size % hp.minibatch_size > 0 else 0)

  for i in range(num_minibatches):
    n = np.minimum((i + 1) * hp.minibatch_size, total_size) - i * hp.minibatch_size
    yield tmp_dataset[i * hp.minibatch_size: i * hp.minibatch_size + n]


def sample_x_init(hp: HParams, n_batch: int = 1) -> np.ndarray:
  s = hp.system()
  res = np.random.uniform(s.bounds[:, 0], s.bounds[:, 1], (n_batch, hp.state_size + hp.control_size))
  res = res[:, :hp.state_size]
  assert np.isfinite(res).all()
  return res
