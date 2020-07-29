import random

import numpy as onp
import simple_parsing

from source.config import Config, HParams
from source.experiment import experiment


if __name__=='__main__':
  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(HParams, dest="hparams")
  parser.add_arguments(Config, dest="config")
  
  args = parser.parse_args()
  hp = args.hparams
  cfg = args.config
  print(hp)
  print(cfg)

  # Set our seeds for reproducibility
  random.seed(hp.seed)
  onp.random.seed(hp.seed)

  # Run experiment
  experiment(hp, cfg)
