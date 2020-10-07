from typing import Callable, Tuple, Optional

import jax.numpy as np
from jax import jit, lax

def integrate(
  dynamics_t: Callable[[np.ndarray, float], np.ndarray], # dynamics function
  x_0: np.ndarray, # starting state
  u: np.ndarray, # controls
  h: float, # step size
  N: int, # steps
) -> Tuple[np.ndarray, np.ndarray]:
  # nh : hold u constant for each integration step (zero-order interpolation)
  @jit
  def rk4_step(x_t1, u):
    k1 = dynamics_t(x_t1, u)
    k2 = dynamics_t(x_t1 + h*k1/2, u)
    k3 = dynamics_t(x_t1 + h*k2/2, u)
    k4 = dynamics_t(x_t1 + h*k3, u)
    return x_t1 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

  fn = lambda x_t, idx: [rk4_step(x_t, u[idx])] * 2
  x_T, ys = lax.scan(fn, x_0, np.arange(N))
  return x_T, np.concatenate((x_0[None], ys))

def integrate_v2( # TODO: if dynamic ODE has a dependancy on t, need to modify rk4_step
  dynamics_t: Callable[[np.ndarray, float], np.ndarray], # dynamics function
  x_0: np.ndarray, # starting state
  u: np.ndarray, # controls
  h: float, # step size
  N: int, # steps
  v: Optional[np.ndarray] = None,
  forward: bool = True, # Forward or Backward mode  #TODO: remove this boolean, and just check if h is pos or neg
) -> Tuple[np.ndarray, np.ndarray]:
  # nh : hold u constant for each integration step (zero-order interpolation)
  @jit
  def rk4_step(x_t1, u, u_next, v, v_next):
    u_convex_approx = (u + u_next)/2
    if forward: hh = h
    else: hh = -1 * h

    # if not np.isnan(v):
    v_convex_approx = (v + v_next) / 2
    k1 = dynamics_t(x_t1, u, v)
    k2 = dynamics_t(x_t1 + hh * k1 / 2, u_convex_approx, v_convex_approx)
    k3 = dynamics_t(x_t1 + hh * k2 / 2, u_convex_approx, v_convex_approx)
    k4 = dynamics_t(x_t1 + hh * k3, u_next, v_next)

    # else:
    # k1 = dynamics_t(x_t1, u)
    # k2 = dynamics_t(x_t1 + hh*k1/2, u_convex_approx)
    # k3 = dynamics_t(x_t1 + hh*k2/2, u_convex_approx)
    # k4 = dynamics_t(x_t1 + hh*k3, u_next)
    return x_t1 + (hh/6)*(k1 + 2*k2 + 2*k3 + k4)

  if v is None:
    v = np.empty_like(u)

  if forward:
    fn = lambda x_t, idx: [rk4_step(x_t, u[idx], u[idx + 1], v[idx], v[idx + 1])] * 2
    x_T, ys = lax.scan(fn, x_0, np.arange(N))
    return x_T, np.concatenate((x_0[None], ys))
  else:
    fn = lambda x_t, idx: [rk4_step(x_t, u[idx], u[idx - 1], v[idx], v[idx - 1])] * 2
    x_T, ys = lax.scan(fn, x_0, np.arange(N,0,-1))
    return x_T, np.concatenate((np.flip(ys), x_0[None]))