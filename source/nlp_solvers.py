import jax.numpy as jnp
from jax import lax
from jax import config
from jax import jit, grad, jacobian, hessian
config.update("jax_enable_x64", True)
from collections import namedtuple
from tensorboardX import SummaryWriter # for parameter tuning

from source.config import SystemType

pre_tuned = {
  SystemType.BACTERIA: {'eta_x': 0.2, 'eta_v': 0.2},  # shooting with 1 interval, 20 controls
  SystemType.CANCER: {'eta_x': 0.1, 'eta_v': 0.1},  # shooting with 1 interval, 30 controls
  SystemType.MOLDFUNGICIDE: {'eta_x': 0.02, 'eta_v': 0.1},  # shooting with 1 interval, 20 controls
  SystemType.GLUCOSE: {'eta_x': 0.1, 'eta_v': 0.1},  # shooting with 1 interval, 10 controls
  # SystemType.BEARPOPULATIONS: {'eta_x': 0.001, 'eta_v': 0.001},  # shooting with 1 interval, 20 controls
  # SystemType.HIVTREATMENT: {'eta_x': 0.01, 'eta_v': 0.01},  # shooting with 1 interval, 60 controls
}


# NOTE: need to tune eta_x and eta_v for each system
def extra_gradient(fun, x0, constraints, bounds, options, system_type):
  """
  Implementation of extragradient method for finding Lagrangian fixed points.

  :param fun: Objective function
  :param x0: Start state
  :param constraints: Equality constraint violation function
  :param bounds: Bounds for decision variables
  :param options: Additional solver options, such as stepsize for primal and dual variable update
  :param system_type: The system under study. Used to set default hyperparameter values.
  :return: A solution dict, containing the values of primal and dual variables,
            whether or not it exited with success, and the value of the objective function at the solution.
  """
  constraint_fun = constraints['fun']
  max_iter = options['maxiter'] if 'maxiter' in options else 30_000
  max_iter = 30_000
  eta_x = options['eta_x'] if 'eta_x' in options else .1  # primals
  eta_v = options['eta_v'] if 'eta_v' in options else .1  # duals
  # atol = options['atol'] if 'atol' in options else 1e-8 # convergence tolerance

  if system_type in pre_tuned:
    eta_x = pre_tuned[system_type]['eta_x']
    eta_v = pre_tuned[system_type]['eta_v']

  @jit
  def lagrangian(x, lmbda):
    """
    :param x: Primals
    :param lmbda: Duals
    :return: Lagrangian
    """
    return fun(x) + lmbda @ constraint_fun(x)

  # Use this for debugging
  def scan(f, init, xs, length=None):
    if xs is None:
      xs = [None] * length
    carry = init
    ys = []
    for x in xs:
      carry, y = f(carry, x)
      ys.append(y)
    return carry, jnp.stack(ys)

  @jit
  def step(x, lmbda):
    """
    Take one step of extragradient.

    :param x: Primals
    :param lmbda: Duals
    :return: (next x, next lmbda)
    """
    x_bar = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x, lmbda),
      a_min=bounds[:, 0], a_max=bounds[:, 1])
    lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(x, lmbda)
    x_new = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
      a_min=bounds[:, 0], a_max=bounds[:, 1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)

    # Use this for debugging, in conjunction with non-jax functions
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

  # harder to debug, but might be faster in some cases
  # def solve(x, lmbda):
  #   def f(val, i):
  #     del i
  #     x, lmbda = val
  #     return step(x, lmbda), None
  #
  #   val, _ = lax.scan(f, (x, lmbda), None, length=max_iter)  # faster
  #   # val, _ = scan(f, (x, lmbda), None, length=max_iter)  # for debugging
  #   x, lmbda = val
  #   return x, lmbda, False

  # For debugging
  def solve(x, lmbda):
    success = False
    x_old = x + 20  # just so we don't terminate immediately
    for i in range(max_iter):
      if i % 1000 == 0:
        print("x", x)
        if i: print("lmbda", lmbda)
      if i % 100 and jnp.allclose(x_old, x, rtol=0., atol=1e-5):  # tune tolerance according to need
        success = True
        break
      x_old = x
      x, lmbda = step(x, lmbda)
    return x, lmbda, success

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda, success = solve(x0, lmbda_init)

  return namedtuple('solution', ['x', 'v', 'success', 'fun'])(x, lmbda, success, fun(x))