from typing import Callable, Tuple

import jax.numpy as np
from jax import jit, lax


def integrate(
  dynamics_t: Callable[[np.ndarray, float], np.ndarray], # dynamics function
  x_0: np.ndarray, # starting state
  u: np.ndarray, # controls
  h: float, # step size
  N: int, # steps
) -> Tuple[np.ndarray, np.ndarray]:
  @jit
  def rk4_step(x_t1, u_t1, u_t2):
    u_mid = (u_t1 + u_t2) / 2
    k1 = dynamics_t(x_t1, u_t1)
    k2 = dynamics_t(x_t1 + h*k1/2, u_mid)
    k3 = dynamics_t(x_t1 + h*k2/2, u_mid)
    k4 = dynamics_t(x_t1 + h*k3, u_t2)
    return x_t1 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

  fn = lambda x_t, idx: [rk4_step(x_t, u[idx], u[idx+1])] * 2
  x_T, ys = lax.scan(fn, x_0, np.arange(N))
  return x_T, np.concatenate((x_0[None], ys))
