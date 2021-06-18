# (c) 2021 Nikolaus Howe
import gin
import jax
import jax.numpy as jnp
import numpy as np
import simple_parsing

from absl import app, flags
from jax.flatten_util import ravel_pytree
from jax.config import config

from myriad.config import Config, HParams, IntegrationOrder, OptimizerType
from myriad.optimizers import get_optimizer
from myriad.plotting import plot
from myriad.utils import integrate

config.update("jax_enable_x64", True)


# Prepare experiment settings
# TODO: Migrate to only using a single parsing technique
parser = simple_parsing.ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")
parser.add_argument("--gin_bindings", type=str)  # Needed for the parser to work in conjunction with absl.flags

key_dict = HParams.__dict__.copy()
key_dict.update(Config.__dict__)
for key in key_dict.keys():
  if "__" not in key:
    flags.DEFINE_string(key, None,  # Parser arguments need to be accepted by the flags
                        'Backward compatibility with previous parser')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "Lab1.A=1.0").')

FLAGS = flags.FLAGS


def plot_zero_control_dynamics(hp, cfg):
  system = hp.system()
  optimizer = get_optimizer(hp, cfg, system)
  num_steps = hp.intervals * hp.controls_per_interval
  stepsize = system.T / num_steps
  zero_us = jnp.zeros((num_steps + 1,))

  _, opt_x = integrate(system.dynamics, system.x_0, zero_us,
                       stepsize, num_steps, None, hp.order)

  plot(hp, system,
       data={'x': opt_x, 'u': zero_us},
       labels={'x': 'Integrated state',
               'u': 'Zero controls'})

  xs_and_us, unused_unravel = ravel_pytree((opt_x, zero_us))
  if hp.optimizer != OptimizerType.FBSM:
    print("control cost from optimizer", optimizer.objective(xs_and_us))
    print('constraint violations from optimizer', jnp.linalg.norm(optimizer.constraints(xs_and_us)))


# Script for running the standard trajectory optimization
def run_trajectory_opt(hp, cfg):
  system = hp.system()
  optimizer = get_optimizer(hp, cfg, system)
  solution = optimizer.solve()
  x = solution['x']
  u = solution['u']
  if hp.order == IntegrationOrder.QUADRATIC and hp.optimizer == OptimizerType.COLLOCATION:
    x_mid = solution['x_mid']
    u_mid = solution['u_mid']
  if optimizer.require_adj:
    adj = solution['adj']

  # TODO: figure out what happens in the quadratic case for integration (how many timesteps does it use in HS vs shooting)

  num_steps = hp.intervals * hp.controls_per_interval
  stepsize = system.T / num_steps

  print("the shapes of x and u are", x.shape, u.shape)

  # times = jnp.linspace(0, system.T, num_steps+1)
  #
  # def dynamics_wrapper(x, t, u):
  #   print("t is", t, ", index is", int(t / stepsize))
  #   single_u = u[int(t / stepsize)]
  #   print("input u", single_u)
  #   return system.dynamics(x, single_u)
  #
  # state_trajectory = odeint(dynamics_wrapper, system.x_0, times, (u,))
  # opt_x = state_trajectory

  _, opt_x = integrate(system.dynamics, system.x_0, u,
                       stepsize, num_steps, None, hp.order)

  if cfg.plot_results:
    if optimizer.require_adj:
      plot(hp, system,
           data={'x': x, 'u': u, 'adj': adj, 'other_x': opt_x},
           labels={'x': ' (from solver)',
                   'u': 'Controls from solver',
                   'adj': 'Adjoint from solver',
                   'other_x': ' (from integrating controls from solver)'})
    else:
      plot(hp, system,
           data={'x': x, 'u': u, 'other_x': opt_x},
           labels={'x': ' (from solver)',
                   'u': 'Controls from solver',
                   'other_x': ' (from integrating controls from solver)'})

  if hp.order == IntegrationOrder.QUADRATIC and hp.optimizer == OptimizerType.COLLOCATION:
    xs_and_us, unused_unravel = ravel_pytree((x, x_mid, u, u_mid))
  else:
    xs_and_us, unused_unravel = ravel_pytree((x, u))
  if hp.optimizer != OptimizerType.FBSM:
    print("control cost from optimizer", optimizer.objective(xs_and_us))
    print('constraint violations from optimizer', jnp.linalg.norm(optimizer.constraints(xs_and_us)))


def main(unused_argv):
  """Main method.
    Args:
      unused_argv: Arguments (unused).
    """
  jax.config.update("jax_enable_x64", True)

  args = parser.parse_args()
  hp = args.hparams
  cfg = args.config
  print(hp)
  print(cfg)

  # Set our seeds for reproducibility
  np.random.seed(hp.seed)

  # Load config, then build system
  gin_files = ['./source/gin-configs/default.gin']
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)

  # -----------------------------------------------------------------------
  ######################
  # Place scripts here #
  ######################
  # system = hp.system()
  # optimizer = get_optimizer(hp, cfg, system)
  # # solution = optimizer.solve()
  # num_steps = hp.intervals * hp.controls_per_interval
  # stepsize = system.T / num_steps
  # u = jnp.sin(jnp.arange(num_steps + 1) * 0.08)
  # # u = jnp.zeros(num_steps + 1)
  # _, opt_x = integrate(system.dynamics, system.x_0, u,
  #                      stepsize, num_steps, None, hp.order)
  # plot(hp, system,
  #      data={'x': opt_x, 'u': u},
  #      labels={'x': ' (from solver)',
  #              'u': 'Controls from solver'})

  # plot_zero_control_dynamics(hp, cfg)
  #
  run_trajectory_opt(hp, cfg)
  # raise SystemExit

  # run_trajectory_opt(hp, cfg)
  # -----------------------------------------------------------------------


if __name__ == '__main__':
  app.run(main)
