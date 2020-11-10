import random

import jax
import numpy as onp
import simple_parsing
import gin

from absl import app
from absl import flags

from source.config import Config, HParams, MParams
from source.optimizers import get_optimizer
from source.systems import get_system

#Prepare experiment settings   #TODO: Use only 1 parsing technique?
parser = simple_parsing.ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")
parser.add_arguments(MParams, dest="syst")
parser.add_argument("--gin_bindings", type=str)  #Needed for the parser to work in conjonction to absl.flags

key_dict = HParams.__dict__.copy()
key_dict.update(Config.__dict__)
for key in key_dict.keys():
  if "__" not in key:
    flags.DEFINE_string(key, None,    #Parser arguments need to be accepted by the flags
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
  hp = args.hparams
  cfg = args.config
  syst = args.syst
  print(hp)
  print(cfg)
  print(syst)

  # Set our seeds for reproducibility
  random.seed(hp.seed)
  onp.random.seed(hp.seed)

  # Run experiment
  gin_files = ['source/configs/default.gin']
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)
  system = get_system(hp)
  #system.update(syst) # Reinitialize  # TODO: remove after fininshing integration og gin-config
  optimizer = get_optimizer(hp, cfg, system)
  x, u, adj = optimizer.solve()  # TODO: accommodate for when solve does not return an adjoint (direct methods)

  if cfg.plot_results:
    system.plot_solution(x, u, adj)

if __name__=='__main__':
  app.run(main)