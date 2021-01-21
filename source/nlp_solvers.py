import jax.numpy as jnp
from jax import lax
from jax import config
from jax import jit, grad, jacobian, hessian
config.update("jax_enable_x64", True)
import sys
from collections import namedtuple
from functools import partial
import time

from source.config import NLPSolverType, HParams

# NOTE: eta requires tuning for each system. Right now, it's an ok speed 
#       for van der pol, but slow for everything else
def extra_gradient(fun, x0, method, constraints, bounds, jac, options):
  del method
  del jac

  constraint_fun = constraints['fun']
  max_iter = options['maxiter'] if 'maxiter' in options else 30_000
  eta_x = options['eta_x'] if 'eta_x' in options else 1e-6 # primals
  eta_v = options['eta_v'] if 'eta_v' in options else 1e-6 # duals
  atol = options['atol'] if 'atol' in options else 1e-6 # convergence tolerance

  # will deal with bounds later
  @jit
  def lagrangian(x, lmbda):
    return fun(x) + lmbda @ constraint_fun(x)

  @jit
  def step(x, lmbda):
    x_bar = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x, lmbda),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(x, lmbda)
    x_new = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)
    return (x_new, lmbda_new)

  def solve(x, lmbda):
    success = False
    x_old = x + 20 # just so we don't terminate immediately
    for i in range(max_iter):
      if i % 1000 == 0:
        print("x", x)
      if i % 100 and jnp.allclose(x_old, x, rtol=0., atol=1e-10): # tune tolerance according to need
        success = True
        break
      x_old = x
      x, lmbda = step(x, lmbda)
    return x, lmbda, success

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda, success = solve(x0, lmbda_init)

  return namedtuple('solution', ['x', 'v', 'success'])(x, lmbda, success)