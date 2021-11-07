# (c) 2021 Nikolaus Howe
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle as pkl

from jax import lax
from typing import Callable, Tuple

from myriad.custom_types import State, States, Control, Controls, DState
from myriad.utils import integrate, integrate_time_independent, integrate_time_independent_in_parallel
from myriad.config import HParams, Config, IntegrationMethod

################
# INSTRUCTIONS #
################
# Place me at the same level as "run.py",
# and run me as:
# for st in SystemType:
  #   if st in [SystemType.SIMPLECASE, SystemType.INVASIVEPLANT]:
  #     continue
  #   print("system", st)
  #   hp.system = st
  #   run_trajectory_opt(hp, cfg)


def nice_scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    for c in carry.T:
      plt.plot(x, c, 'o', color='red')
    # if x == 21:
    #   print("x", x, carry)
    if x == 49:
      print("xx", x, carry)
      # plt.xlim((0, 12))
      plt.show()
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)


def testing_integrate_time_independent(
        dynamics_t: Callable[[State, Control], DState],  # dynamics function
        x_0: State,  # starting state
        interval_us: Controls,  # controls
        h: float,  # step size
        N: int,  # steps
        integration_method: IntegrationMethod  # allows user to choose int method
) -> Tuple[State, States]:
  # QUESTION: do we want to keep the mid-controls as decision variables for RK4,
  # or move to simply taking the average between the edge ones?
  # @jit
  def rk4_step(x, u1, u2, u3):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h * k1 / 2, u2)
    k3 = dynamics_t(x + h * k2 / 2, u2)
    k4 = dynamics_t(x + h * k3, u3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  # @jit
  def heun_step(x, u1, u2):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h * k1, u2)
    return x + h / 2 * (k1 + k2)

  # @jit
  def midpoint_step(x, u1, u2):
    x_mid = x + h * dynamics_t(x, u1)
    u_mid = (u1 + u2) / 2
    return x + h * dynamics_t(x_mid, u_mid)

  # @jit
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

  # x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  plt.plot(interval_us, color='blue')
  x_T, all_next_states = nice_scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


def probe(hp: HParams, cfg: Config):
  hp.key, subkey = jax.random.split(hp.key)

  system = hp.system()

  # Generate |total dataset size| control trajectories
  total_size = hp.train_size + hp.val_size + hp.test_size

  state_size = system.x_0.shape[0]
  control_size = system.bounds.shape[0] - state_size

  u_lower = system.bounds[state_size:, 0]
  u_upper = system.bounds[state_size:, 1]
  x_lower = system.bounds[:state_size, 0]
  x_upper = system.bounds[:state_size, 1]
  if jnp.isinf(u_lower).any() or jnp.isinf(u_upper).any():
    raise Exception("infinite control bounds, aborting")
  # if jnp.isinf(x_lower).any() or jnp.isinf(x_upper).any():
  #   raise Exception("infinite state bounds, aborting")

  spread = (u_upper - u_lower) * hp.sample_spread

  all_us = jax.random.uniform(subkey, (total_size, hp.num_steps + 1, control_size),
                              minval=u_lower, maxval=u_upper)

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

  print("the shape of the generated us is", all_us.shape)
  print("the shape of the generated xs is", all_xs.shape)
  # print("an example is", all_xs[-1])

  for i, xs in enumerate(all_xs):
    if not jnp.isfinite(xs).all():
      print("there was an infinity encountered")
      print("us", all_us[i])
      print("start state", start_states[i])
      raise SystemExit

  plt.close()
  for i, xs in enumerate(all_xs):
    # print("xs is of shape", xs.shape)
    for j, state in enumerate(xs.T):
      plt.plot(state)

    # break
    # plt.plot(xs)
  # plt.legend()
  plt.show()
  # plt.savefig(f'cool_{hp.system.name}.pdf')
  plt.close()


def special_probe(hp, cfg):
  # CARTPOLE
  key = jax.random.PRNGKey(42)
  key, subkey = jax.random.split(key)

  system = hp.system()
  hp.key, subkey = jax.random.split(hp.key)

  file_path = 't_set'
  train_set = pkl.load(open(file_path, 'rb'))
  print("train set", train_set)

  first_xs = train_set[0, :, :hp.state_size]
  first_us = train_set[0, :, hp.state_size:]

  given_params = {
        'g': 15,
        'm1': 1.0,
        'm2': 0.1,
        'length': 1.0
      }
  def dynamics(x, u):
    return system.parametrized_dynamics(given_params, x, u)

  print("first xs", first_xs.shape)
  print("first us", first_us.shape)

  start = first_xs[0]
  print("start", start.shape)

  _, xs = testing_integrate_time_independent(dynamics, start,
                                             first_us, hp.stepsize, hp.num_steps,
                                             hp.integration_method)

  #####################
  # train_xs = train_set[:, :, :hp.state_size]
  # train_us = train_set[:, :, hp.state_size:]
  # start_xs = train_xs[:, 0, :]
  # # if cfg.verbose:
  # #   print("train xs", train_xs.shape)
  # #   print("train us", train_us.shape)
  # #   print("start train xs", start_xs.shape)
  #
  # _, predicted_states = integrate_time_independent_in_parallel(
  #   dynamics, start_xs, train_us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
  # )
  ############################3

  print("us", first_us)
  print("resulting xs", xs)
  raise SystemExit

  # print("us", us)
  # print("xs", xs)
  u = us[21]
  uu = us[22]
  uuu = us[23]

  x = jnp.array([-2.68629165])
  xx = jnp.array([-7.63482766])

  def heun_step(x, u1, u2):
    k1 = system.dynamics(x, u1)
    print("k1", k1)
    k2 = system.dynamics(x + hp.stepsize * k1, u2)
    print("k2", k2)
    return x + hp.stepsize / 2 * (k1 + k2)

  print("step", heun_step(xx, uu, uuu))
