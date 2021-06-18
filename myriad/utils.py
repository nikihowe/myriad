from __future__ import annotations

import jax
import jax.numpy as jnp
import time
import functools
import typing

from jax.flatten_util import ravel_pytree

from jax import jit, lax, vmap
from scipy.optimize import minimize
from typing import Callable, Optional, Union, Tuple, Dict

from myriad.config import IntegrationOrder, HParams, NLPSolverType, OptimizerType
from myriad.systems import FiniteHorizonControlSystem
# from source.optimizers import get_optimizer
from myriad.nlp_solvers import extra_gradient
from myriad.plotting import plot
from myriad.types import Control, Controls, State, States, Cost, Time


def integrate(
  dynamics_t: Callable[[jnp.ndarray, Control, Time], jnp.ndarray],  # dynamics function
  x_0: jnp.ndarray,           # starting state
  interval_us: jnp.ndarray,   # controls
  h: float,                   # step size
  N: int,                     # steps
  ts: Optional[jnp.ndarray],  # allow for optional time-dependent dynamics
  integration_order: IntegrationOrder = IntegrationOrder.LINEAR,  # allows user to choose interpolation for controls
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  # QUESTION: do we want to keep this interpolation for rk4, or move to linear?
  @jit
  def rk4_step(x, u1, u2, u3, t=0):
    k1 = dynamics_t(x, u1, t)
    k2 = dynamics_t(x + h*k1/2, u2, t + h/2)
    k3 = dynamics_t(x + h*k2/2, u2, t + h/2)
    k4 = dynamics_t(x + h*k3, u3, t + h)
    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

  @jit
  def heun_step(x, u1, u2, t=0):
    k1 = dynamics_t(x, u1, t)
    k2 = dynamics_t(x + h*k1, u2, t + h/2)
    return x + (h/2) * (k1 + k2)

  @jit
  def euler_step(x, u, t=0):
    return x + h*dynamics_t(x, u, t)

  def fn(carried_state, idx):
    nonlocal integration_order
    if not integration_order:
      integration_order = IntegrationOrder.CONSTANT
    if integration_order == IntegrationOrder.CONSTANT:
      if ts is not None:
        one_step_forward = euler_step(carried_state, interval_us[idx], ts[idx])
      else:
        one_step_forward = euler_step(carried_state, interval_us[idx])
    elif integration_order == IntegrationOrder.LINEAR:
      if ts is not None:
        one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1], ts[idx])
      else:
        one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1])
    elif integration_order == IntegrationOrder.QUADRATIC:
      if ts is not None:
        one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1],
                                    interval_us[2*idx+2], ts[idx])
      else:
        one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2])
    else:
      print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
      raise KeyError

    return one_step_forward, one_step_forward  # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


# Used for the augmented state cost calculation
integrate_in_parallel = vmap(integrate, in_axes=(None, 0, 0, None, None, 0, None))  # , static_argnums=(0, 5, 6)


def integrate_time_independent(
  dynamics_t: Callable[[jnp.ndarray, Control], jnp.ndarray],  # dynamics function
  x_0: jnp.ndarray,           # starting state
  interval_us: jnp.ndarray,   # controls
  h: float,                   # step size
  N: int,                     # steps
  integration_order: IntegrationOrder = IntegrationOrder.LINEAR,  # allows user to choose interpolation for controls
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  # QUESTION: do we want to keep this interpolation for rk4, or move to linear?
  @jit
  def rk4_step(x, u1, u2, u3):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h*k1/2, u2)
    k3 = dynamics_t(x + h*k2/2, u2)
    k4 = dynamics_t(x + h*k3, u3)
    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

  @jit
  def heun_step(x, u1, u2):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h*k1, u2)
    return x + (h/2) * (k1 + k2)

  @jit
  def euler_step(x, u):
    return x + h*dynamics_t(x, u)

  def fn(carried_state, idx):
    nonlocal integration_order
    if not integration_order:
      integration_order = IntegrationOrder.CONSTANT
    if integration_order == IntegrationOrder.CONSTANT:
      one_step_forward = euler_step(carried_state, interval_us[idx])
    elif integration_order == IntegrationOrder.LINEAR:
      one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1])
    elif integration_order == IntegrationOrder.QUADRATIC:
      one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2])
    else:
      print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
      raise KeyError

    return one_step_forward, one_step_forward  # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


integrate_time_independent_in_parallel = vmap(integrate_time_independent, in_axes=(None, 0, 0, None, None, None))


# Used for the adjoint integration
def integrate_fbsm(
  dynamics_t: Callable[[jnp.ndarray, Union[float, jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]],
                       jnp.ndarray],  # dynamics function
  x_0: jnp.ndarray,                   # starting state
  u: jnp.ndarray,                     # controls
  h: float,                           # step size  # is negative in backward mode
  N: int,                             # steps
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
    u_convex_approx = (u + u_next)/2
    v_convex_approx = (v + v_next)/2

    k1 = dynamics_t(x_t1, u, v, t)
    k2 = dynamics_t(x_t1 + h * k1/2, u_convex_approx, v_convex_approx, t + h/2)
    k3 = dynamics_t(x_t1 + h * k2/2, u_convex_approx, v_convex_approx, t + h/2)
    k4 = dynamics_t(x_t1 + h * k3, u_next, v_next, t + h)

    return x_t1 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

  if v is None:
    v = jnp.empty_like(u)
  if t is None:
    t = jnp.empty_like(u)

  direction = int(jnp.sign(h))
  if discrete:
    if direction >= 0:
      fn = lambda x_t, idx: [dynamics_t(x_t, u[idx], v[idx], t[idx])] * 2
    else:
      fn = lambda x_t, idx: [dynamics_t(x_t, u[idx], v[idx-1], t[idx-1])] * 2
  else:
    fn = lambda x_t, idx: [rk4_step(x_t, u[idx], u[idx + direction], v[idx], v[idx + direction], t[idx])] * 2
  if direction >= 0:
    x_T, ys = lax.scan(fn, x_0, jnp.arange(N))
    return x_T, jnp.concatenate((x_0[None], ys))

  else:
    x_T, ys = lax.scan(fn, x_0, jnp.arange(N, 0, -1))
    return x_T, jnp.concatenate((jnp.flipud(ys), x_0[None]))

# TODO: make the start state default to the system start state
def get_state_trajectory_and_cost(hp: HParams, system: FiniteHorizonControlSystem,
                                  start_state: State, us: Controls) -> Tuple[States, Cost]:
  @jax.jit
  def augmented_dynamics(x_and_c: jnp.ndarray, u: float, t: jnp.ndarray) -> jnp.ndarray:
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
    step_size, num_steps, times, hp.order)

  # print("the states and costs are", state_and_cost)
  # print("the states and costs are", state_and_cost.shape)
  # raise SystemExit

  states = state_and_cost[:, :-1]

  # print("extracted states", states.shape)

  last_augmented_state = state_and_cost[-1]
  # print("last aug state", last_augmented_state)
  cost = last_augmented_state[-1]
  print("cost", cost)
  if system.terminal_cost:
    cost += system.terminal_cost_fn(last_augmented_state[:-1], us[-1])
  # raise SystemExit
  return states, cost


def solve(hp, cfg, opt_dict) -> Dict[str, jnp.ndarray]:
  """
  Use a the solver indicated in the hyper-parameters to solve the constrained optimization problem.
  Args:
    hp: the hyperparameters
    cfg: the extra hyperparameters
    opt_dict: everything needed for the solve

  Returns
    A dictionary with the optimal controls and corresponding states (and for quadratic interpolation schemes, the midpoints too)
  """
  _t1 = time.time()
  opt_inputs = {
    'fun': jax.jit(opt_dict['objective']) if cfg.jit else opt_dict['objective'],
    'x0': opt_dict['guess'],
    'constraints': ({
      'type': 'eq',
      'fun': jax.jit(opt_dict['constraints']) if cfg.jit else opt_dict['constraints'],
      'jac': jax.jit(jax.jacrev(opt_dict['constraints'])) if cfg.jit else jax.jacrev(opt_dict['constraints']),
    }),
    'bounds': opt_dict['bounds'],
    'jac': jax.jit(jax.grad(opt_dict['objective'])) if cfg.jit else jax.grad(opt_dict['objective']),
    'options': {"maxiter": hp.max_iter}
  }
  if hp.nlpsolver == NLPSolverType.EXTRAGRADIENT:
    opt_inputs['method'] = 'exgd'
    solution = extra_gradient(**opt_inputs, system_type=hp.system)
  elif hp.nlpsolver == NLPSolverType.SLSQP:
    opt_inputs['method'] = 'SLSQP'
    solution = minimize(**opt_inputs)
  elif hp.nlpsolver == NLPSolverType.TRUST:
    opt_inputs['method'] = 'trust-constr'
    solution = minimize(**opt_inputs)
  elif hp.nlpsolver == NLPSolverType.IPOPT:
    print("haven't installed ipopt yet, using Scipy")
    # solution = minimize_ipopt(**opt_inputs)
    opt_inputs['method'] = 'SLSQP'
    solution = minimize(**opt_inputs)
  else:
    print("Unknown NLP solver. Please choose among", list(NLPSolverType.__members__.keys()))
    raise ValueError
  _t2 = time.time()
  if cfg.verbose:
    print('Solver exited with success:', solution.success)
    print(f'Completed in {_t2 - _t1} seconds.')
    print('Cost at solution:', solution.fun)

  results = {}
  if hp.order == IntegrationOrder.QUADRATIC and hp.optimizer == OptimizerType.COLLOCATION:
    results['x'], results['x_mid'], results['u'], results['u_mid'] = opt_dict['unravel'](solution.x)
  else:
    results['x'], results['u'] = opt_dict['unravel'](solution.x)

  return results
