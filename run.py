import random

import jax
import numpy as np
import simple_parsing
import gin
from jax.flatten_util import ravel_pytree

from absl import app
from absl import flags

from source.config import Config, HParams, OptimizerType
from source.optimizers import get_optimizer
from source.plotting import plot_result
from source.utils import integrate


# Prepare experiment settings   # TODO: Use only 1 parsing technique?
parser = simple_parsing.ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")
parser.add_argument("--gin_bindings", type=str)  # Needed for the parser to work in conjonction to absl.flags

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


def main(unused_argv):
  """Main method.
    Args:
      unused_argv: Arguments (unused).
    """
  jax.config.update("jax_enable_x64", True)

  args = parser.parse_args()
  hp: HParams = args.hparams
  cfg: Config = args.config
  print(hp)
  print(cfg)

  # Set our seeds for reproducibility
  random.seed(hp.seed)
  np.random.seed(hp.seed)

  # Load config, then build system
  gin_files = ['./source/gin-configs/default.gin']
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)
  system = hp.system()

  # Run experiment
  optimizer = get_optimizer(hp, cfg, system)
  results = optimizer.solve()
  # print("results", results)
  # print('integrated cost:', optimizer.objective(ravel_pytree(results)))
  if cfg.plot_results:
    plot_result(results, hp)

  # Check how good the run was
  if hp.optimizer == OptimizerType.FBSM:
    x, u, adj = results
    print("xs", x.shape)
    print("us", u.shape)
    hp.intervals = 1
    hp.controls_per_interval = hp.fbsm_intervals
    hp.optimizer = OptimizerType.SHOOTING
    new_optimizer = get_optimizer(hp, cfg, system)
    print("integrated cost", new_optimizer.objective(ravel_pytree((x, u))))



if __name__ == '__main__':
  app.run(main)