import jax.numpy as jnp
from jax import lax
from jax import config
from jax import jit, grad, jacobian, hessian
config.update("jax_enable_x64", True)
from collections import namedtuple
from tensorboardX import SummaryWriter # for parameter tuning

# NOTE: need to tune eta_x and eta_v for each system
def extra_gradient(fun, x0, method, constraints, bounds, jac, options):
  del method
  del jac

  constraint_fun = constraints['fun']
  max_iter = options['maxiter'] if 'maxiter' in options else 30_000
  eta_x = options['eta_x'] if 'eta_x' in options else 1 # primals
  eta_v = options['eta_v'] if 'eta_v' in options else 1e-1 # duals
  # atol = options['atol'] if 'atol' in options else 1e-8 # convergence tolerance

  @jit
  def lagrangian(x, lmbda):
    return fun(x) + lmbda @ constraint_fun(x)

  def step(x, lmbda):
    x_bar = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x, lmbda),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(x, lmbda)
    x_new = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
      a_min=bounds[:,0], a_max=bounds[:,1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)

    if jnp.isnan(x_new).any() or jnp.isnan(lmbda_new).any():
      print("WE GOT NANS")
      print("x", x)
      print("lmbda", lmbda)
      print("xbar", x_bar)
      print("xnew", x_new)
      print("lmbda_bar", lmbda_bar)
      print("lmbda_new", lmbda_new)
      print("grad_x at start", grad(lagrangian, argnums=0)(x, lmbda))
      print("grad_lmbda at start", grad(lagrangian, argnums=1)(x, lmbda))
      print("grad_x at bar", grad(lagrangian, argnums=0)(x_bar, lmbda_bar))
      print("grad_lmbda at bar", grad(lagrangian, argnums=1)(x_bar, lmbda_bar))
      raise SystemExit

    return x_new, lmbda_new

  @jit
  def solve(x, lmbda):
    def f(val, i):
      del i
      x, lmbda = val
      return step(x, lmbda), None

    val, _ = lax.scan(f, (x, lmbda), None, length=max_iter)
    x, lmbda = val
    return x, lmbda, False

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda, success = solve(x0, lmbda_init)

  return namedtuple('solution', ['x', 'v', 'success'])(x, lmbda, success)