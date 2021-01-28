import jax.numpy as jnp
from jax import lax
from jax import config
from jax import jit, grad, jacobian, hessian
config.update("jax_enable_x64", True)
import sys
from collections import namedtuple
from functools import partial
import time
from tensorboardX import SummaryWriter # for parameter tuning

from source.config import NLPSolverType, HParams

writer = SummaryWriter()

# NOTE: eta requires tuning for each system. Right now, it's an ok speed 
#       for van der pol, but slow for everything else
def extra_gradient(fun, x0, method, constraints, bounds, jac, options):
  del method
  del jac

  constraint_fun = constraints['fun']
  max_iter = options['maxiter'] if 'maxiter' in options else 30_000
  eta_x = options['eta_x'] if 'eta_x' in options else 1 # primals
  eta_v = options['eta_v'] if 'eta_v' in options else 1e-1 # duals
  atol = options['atol'] if 'atol' in options else 1e-8 # convergence tolerance

  # will deal with bounds later
  @jit
  def lagrangian(x, lmbda):
    return fun(x) + lmbda @ constraint_fun(x)

  @jit
  def step(x, lmbda):
    # print("dimension of x", x.shape)
    # print("dim of lmbda", lmbda.shape)
    # print("bounds", bounds.shape)
    # print("bounds", bounds)
    # raise SystemExit
    # v_bar = 0.9 * v + eta_x * grad(lagrangian, argnums=0)(x, lmbda)
    # x_bar = jnp.clip(x - v_bar, a_min=bounds[:,0], a_max=bounds[:,1])
    # lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(x, lmbda)

    # v_new = 0.9 * v_bar + eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar)
    # x_new = jnp.clip(x - v_new, a_min=bounds[:,0], a_max=bounds[:,1])
    # lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)

    x_bar = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x, lmbda),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(x, lmbda)
    x_new = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)
    
    # if jnp.isnan(x_new).any() or jnp.isnan(lmbda_new).any():
    #   print("WE GOT NANS")
    #   print("x", x)
    #   print("lmbda", lmbda)
    #   print("xbar", x_bar)
    #   print("xnew", x_new)
    #   print("lmbda_bar", lmbda_bar)
    #   print("lmbda_new", lmbda_new)
    #   print("grad_x at start", grad(lagrangian, argnums=0)(x, lmbda))
    #   print("grad_lmbda at start", grad(lagrangian, argnums=1)(x, lmbda))
    #   print("grad_x at bar", grad(lagrangian, argnums=0)(x_bar, lmbda_bar))
    #   print("grad_lmbda at bar", grad(lagrangian, argnums=1)(x_bar, lmbda_bar))
    #   raise SystemExit
    return x_new, lmbda_new

  # @jit
  # def solve(x, lmbda):
  #   def loop_body(i, val):
  #     x, lmbda = val
  #     x, lmbda = step(x, lmbda)
  #     return x, lmbda

  #   x, lmbda = lax.fori_loop(0, max_iter, loop_body, (x, lmbda))
  #   return x, lmbda, False

  # def scan(f, init, xs, length=None):
  #   if xs is None:
  #     xs = [None] * length
  #   carry = init
  #   ys = []
  #   for x in xs:
  #     print("carry", carry)
  #     print("x", x)
  #     carry, y = f(carry, x)
  #     ys.append(y)
  #   return carry, np.stack(ys)

  @jit
  def solve(x, lmbda):
    def f(val, i):
      x, lmbda = val
      return step(x, lmbda), None

    val, _ = lax.scan(f, (x, lmbda), None, length=max_iter)
    x, lmbda = val
    return x, lmbda, False


  # def solve(x, lmbda):
  #   nonlocal eta_x, eta_v # allows to modify the outer values

  #   # v = 0.

  #   success = False
  #   x_old = x + 20 # just so we don't terminate immediately
  #   for i in range(max_iter):
  #     # if i % 100 == 0:
  #       # Tensorboard stuff here
  #       # writer.add_scalar('data/fx', fun(x), i)
  #       # writer.add_scalar('data/hx', jnp.linalg.norm(constraint_fun(x)), i)
  #       # writer.add_scalar('data/lag_x_lmbda', lagrangian(x, lmbda), i)
  #       # # writer.add_scalar('data/eta_x', eta_x, i)
  #       # # writer.add_scalar('data/eta_v', eta_v, i)
  #       # for d, hi in enumerate(constraint_fun(x)):
  #       #   writer.add_scalar('data/hx_{}'.format(d), hi, i)

  #       # print(i)
  #     # if i % 1000 == 0 and (jnp.isnan(x).any() or jnp.isnan(lmbda).any()): # this is a failure case
  #     #   x = jnp.nan_to_num(x)
  #     #   lmbda = jnp.nan_to_num(lmbda)
  #     #   print("nans")
  #     #   break
  #     # if i % 1000 == 0 and jnp.allclose(x_old, x, rtol=0., atol=atol): # tune tolerance according to need
  #     #   success = True
  #     #   break

  #     # Decrease step size
  #     if i % 1000 == 0:
  #       eta_x *= 0.999
  #       eta_v *= 0.999

  #     x_old = x
  #     # lmbda_old = lmbda
  #     # x, v, lmbda = step(x, v, lmbda)
  #     x, lmbda = step(x, lmbda)

  #     # if jnp.isnan(x).any() or jnp.isnan(lmbda).any():
  #     #   print("WE GOT NANS")
  #     #   print("old x", x_old)
  #     #   print("cur x", x)
  #     #   print("old lmbda", lmbda_old)
  #     #   print("cur lmbda", lmbda)
  #     #   raise SystemExit
  #   return x, lmbda, success

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda, success = solve(x0, lmbda_init)

  # writer.export_scalars_to_json("./all_scalars.json")
  # writer.close()
  return namedtuple('solution', ['x', 'v', 'success'])(x, lmbda, success)