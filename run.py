import random

import jax
import jax.numpy as jnp
import numpy as np
import simple_parsing
import gin

from absl import app, flags

from source.config import Config, HParams
from source.config import OptimizerType
from source.optimizers import get_optimizer
from source.systems import get_system
from source.utils import integrate
from source.plotting import plot

# Prepare experiment settings   # TODO: Use only 1 parsing technique?
parser = simple_parsing.ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")
parser.add_argument("--gin_bindings", type=str)  # Needed for the parser to work in conjunction with absl.flags

key_dict = HParams.__dict__.copy()
key_dict.update(Config.__dict__)
for key in key_dict.keys():
  if "__" not in key:
    flags.DEFINE_string(key, None,    # Parser arguments need to be accepted by the flags
                        'Backward compatibility with previous parser')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "Lab1.A=1.0").')

FLAGS = flags.FLAGS

# Load config, then build system


def main(unused_argv):
  """ Main method.
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
  random.seed(hp.seed)
  np.random.seed(hp.seed)

  # TODO: make sure that it's ok for this to live outside main
  gin_files = ['./source/gin-configs/default.gin']
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)

  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)

  # put this in run.py
  if optimizer.require_adj:
    x, u, adj = optimizer.solve()
  else:
    x, u = optimizer.solve()

  if hp.optimizer == OptimizerType.FBSM:
    num_steps = hp.steps
  else:
    num_steps = hp.intervals*hp.controls_per_interval

  stepsize = system.T / num_steps
  _, opt_x = integrate(system.dynamics, system.x_0, u,
                       stepsize, num_steps, None, hp.order)

  if cfg.plot_results:
    if optimizer.require_adj:
      plot(system,
           data={'x': opt_x, 'u': u, 'adj': adj},
           labels={'x': 'Resulting state trajectory',
                   'u': 'Controls from solver',
                   'adj': 'Adjoint from solver'})
    else:
      plot(system,
           data={'x': opt_x, 'u': u},
           labels={'x': 'Resulting state trajectory',
                   'u': 'Controls from solver'})

  xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
  if hp.optimizer != OptimizerType.FBSM:
    print("control cost", optimizer.objective(xs_and_us))
    print('constraint_violations', jnp.linalg.norm(optimizer.constraints(xs_and_us)))
  raise SystemExit

  # -----------------------------------------------------------------------


if __name__ == '__main__':
  app.run(main)
