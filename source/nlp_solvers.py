import jax.numpy as jnp
from jax import lax
from jax import config
from jax import jit, grad, jacobian, hessian
config.update("jax_enable_x64", True)
import sys
from collections import namedtuple
from functools import partial
import time

from .config import NLPSolverType, HParams

# NOTE: eta requires tuning for each system. Right now, it's an ok speed 
#       for van der pol, but slow for everything else
def extra_gradient(fun, x0, method, constraints, bounds, jac, options):
  del method
  del jac

  constraint_fun = constraints['fun']
  max_iter = 10*options['maxiter'] if 'maxiter' in options else 30_000

  # will deal with bounds later
  @jit
  def lagrangian(x, lmbda):
    return fun(x) + lmbda @ constraint_fun(x)

  @jit
  def step(x, lmbda):
    eta = 0.01 # TODO: tune this, and have different ones
               # for the primal and dual variables
    x_bar = jnp.clip(x - eta * grad(lagrangian, argnums=0)(x, lmbda),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_bar = lmbda + eta * grad(lagrangian, argnums=1)(x, lmbda)
    x_new = jnp.clip(x - eta * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_new = lmbda + eta * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)
    return (x_new, lmbda_new)

  def solve(x, lmbda):
    x_old = x + 20 # just so we don't terminate immediately
    for i in range(max_iter):
      if i % 1000 == 0:
        print("x", x)
      if i % 100 and jnp.allclose(x_old, x, rtol=0., atol=1e-5): # tune tolerance according to need
        break
      x_old = x
      x, lmbda = step(x, lmbda)
    return x, lmbda

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda = solve(x0, lmbda_init)

  return namedtuple('solution', ['x', 'v'])(x, lmbda)