import jax.numpy as np
from jax import jit, lax

def integrate_fwd(dynamics_t, y_0, u, h, M):
  @jit
  def rk4_step_fwd(y_t1, u_t1, u_t2):
    u_mid = (u_t1 + u_t2) / 2
    k1 = dynamics_t(y_t1, u_t1)
    k2 = dynamics_t(y_t1 + h*k1/2, u_mid)
    k3 = dynamics_t(y_t1 + h*k2/2, u_mid)
    k4 = dynamics_t(y_t1 + h*k3, u_t2)
    return y_t1 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

  fn = lambda y_t, idx: [rk4_step_fwd(y_t, u[idx], u[idx+1])] * 2
  ys = lax.scan(fn, y_0, np.arange(M))[1]
  return np.concatenate((y_0[None], ys))
