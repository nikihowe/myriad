import random

import jax
import numpy as onp
import simple_parsing
import gin

from source.config import Config, HParams, MParams
from source.optimizers import get_optimizer
from source.systems import get_system


if __name__=='__main__':
  jax.config.update("jax_enable_x64", True)

  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(HParams, dest="hparams")
  parser.add_arguments(Config, dest="config")
  parser.add_arguments(MParams, dest="syst")
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
  gin.parse_config_file('source/configs/default.gin')
  system = get_system(hp)
  #system.update(syst) # Reinitialize  # TODO: remove after fininshing integration og gin-config
  optimizer = get_optimizer(hp, cfg, system)
  x, u, adj = optimizer.solve()  # TODO: accomodate for when solve does not return an adjoint (direct methods)

  #debug tool:
  print(system.A)

  if cfg.plot_results:
    system.plot_solution(x, u, adj)
