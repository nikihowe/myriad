# (c) 2021 Nikolaus Howe
import jax.numpy as jnp

from jax import jit, grad

from tensorboardX import SummaryWriter  # for parameter tuning
writer = SummaryWriter()


def extra_gradient(fun, x0, method, constraints, bounds, jac, options):
  del method, jac

  print("we're trying exgd with steps:", options['maxiter'])

  constraint_fun = constraints['fun']
  max_iter = options['maxiter'] if 'maxiter' in options else 30_000
  eta_x = options['eta_x'] if 'eta_x' in options else 1e-1  # primals
  eta_v = options['eta_v'] if 'eta_v' in options else 1e-3  # duals
  atol = options['atol'] if 'atol' in options else 1e-6  # convergence tolerance

  @jit
  def lagrangian(x, lmbda):
    return fun(x) + lmbda @ constraint_fun(x)

  @jit
  # We address bounds by clipping
  def step(x, lmbda):
    x_bar = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x, lmbda),
                     a_min=bounds[:, 0], a_max=bounds[:, 1])
    x_new = jnp.clip(x - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda),
                     a_min=bounds[:, 0], a_max=bounds[:, 1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_new, lmbda)
    return x_new, lmbda_new

  def solve(x, lmbda):
    nonlocal eta_x, eta_v  # so we can modify them during solve

    success = False
    x_old = x + 20  # just so we don't terminate immediately
    for i in range(max_iter):

      if i % 2000 == 0:
        # Tensorboard recording here
        writer.add_scalar('loss/fx', fun(x), i)
        cur_lag = lagrangian(x, lmbda)
        writer.add_scalar('lagrangian/lag', cur_lag, i)
        for d, hi in enumerate(constraint_fun(x)):
          writer.add_scalar('vars/lambda_{}'.format(d), lmbda[d], i)
          writer.add_scalar('constraints/hx_{}'.format(d), hi, i)

      # Success
      if i % 1000 == 0 and jnp.allclose(x_old, x, rtol=0., atol=atol):  # tune tolerance according to need
        success = True
        break

      # Decrease step size
      if i % 1000 == 0:
        eta_x *= 0.999
        eta_v *= 0.999

      x_old = x
      x, lmbda = step(x, lmbda)

      if i % 1000 and (jnp.isnan(x).any() or jnp.isnan(lmbda).any()):
        print("WE GOT NANS")
        print("cur x", x)
        print("cur lmbda", lmbda)
        raise SystemExit
    writer.close()

    return x, lmbda, success

  lmbda_init = jnp.ones_like(constraint_fun(x0))
  x, lmbda, success = solve(x0, lmbda_init)

  # writer.export_scalars_to_json("./all_scalars.json")
  return {
    'x': x,
    'v': lmbda,
    'fun': fun(x),
    'success': success
  }