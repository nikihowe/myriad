from typing import Callable, Tuple, Optional, Union

import jax.numpy as jnp
from jax import jit, lax, vmap

from source.config import IntegrationOrder


# TODO: make this work for time-dependent dynamics
def integrate(
  dynamics_t: Callable[[jnp.ndarray, float], jnp.ndarray],  # dynamics function
  x_0: jnp.ndarray,           # starting state
  interval_us: jnp.ndarray,   # controls
  h: float,                   # step size
  N: int,                     # steps
  ts: Optional[jnp.ndarray],  # allow for optional time-dependent dynamics
  integration_order: IntegrationOrder = IntegrationOrder.CONSTANT,  # allows user to choose interpolation for controls
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  # QUESTION: do we want to keep this interpolation for rk4, or move to linear?
  @jit
  def rk4_step(x, u1, u2, u3, ts=None):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h*k1/2, u2)
    k3 = dynamics_t(x + h*k2/2, u2)
    k4 = dynamics_t(x + h*k3, u3)
    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

  @jit
  def heun_step(x, u1, u2, ts=None):
    k1 = dynamics_t(x, u1)
    k2 = dynamics_t(x + h*k1, u2)
    return x + (h/2) * (k1 + k2)

  @jit
  def euler_step(x, u, ts=None):
    return x + h*dynamics_t(x, u)

  def fn(carried_state, idx):
    nonlocal integration_order
    if not integration_order:
      integration_order = IntegrationOrder.CONSTANT
    if integration_order == IntegrationOrder.CONSTANT:
      if ts is not None:
        one_step_forward = euler_step(carried_state, interval_us[idx], *ts[idx:idx+2])
      else:
        one_step_forward = euler_step(carried_state, interval_us[idx])
    elif integration_order == IntegrationOrder.LINEAR:
      if ts is not None:
        one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1], *ts[idx:idx+2])
      else:
        one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1])
    elif integration_order == IntegrationOrder.QUADRATIC:
      if ts is not None:
        one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2], *ts[idx:idx+2])
      else:
        one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2])
    else:
      print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
      raise KeyError

    return one_step_forward, one_step_forward # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[jnp.newaxis], all_next_states))


# Used for the augmented state cost calculation
integrate_in_parallel = vmap(integrate, in_axes=(None, 0, 0, None, None, 0, None))


# Used for the adjoint integration
def integrate_v2(
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
  # nh : hold u constant for each integration step (zero-order interpolation)
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
