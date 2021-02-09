import random
from source.systems import get_system
import unittest

import jax
import numpy as np
from absl import flags
import simple_parsing
import gin
from datetime import date

from source.config import Config, SystemType, HParams, OptimizerType, IntegrationOrder
from source.optimizers import get_optimizer
from source.utils import integrate
from source.plotting import plot
from source.systems import IndirectFHCS

jax.config.update("jax_enable_x64", True)

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


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def test_smoke(self):
    # Test all the systems, using collocation for the base ones,
    # and using fbsm for the lenhart ones
    for system in SystemType:
      with self.subTest(system=system):
        args = parser.parse_args()
        hp = args.hparams
        cfg = args.config
        if isinstance(system, IndirectFHCS):
          hp.optimizer = OptimizerType.FBSM
        else:
          hp.optimizer = OptimizerType.COLLOCATION
        cfg = Config(verbose=True, plot_results=True)
        random.seed(hp.seed)
        np.random.seed(hp.seed)

        gin_files = ['./source/gin-configs/default.gin']
        gin_bindings = FLAGS.gin_bindings
        gin.parse_config_files_and_bindings(gin_files,
                                            bindings=gin_bindings,
                                            skip_unknown=False)
        _system = get_system(hp)
        optimizer = get_optimizer(hp, cfg, _system)

        if optimizer.require_adj:
          x, u, adj = optimizer.solve()
        else:
          x, u = optimizer.solve()

        if hp.optimizer == OptimizerType.FBSM:
          num_steps = hp.fbsm_intervals
        else:
          num_steps = hp.intervals * hp.controls_per_interval

        stepsize = system.T / num_steps
        _, opt_x = integrate(system.dynamics, system.x_0, u,
                             stepsize, num_steps, None, hp.order)

        save_as = str(date.today()) + str(hp.system.name)
        title = "TEST"
        if cfg.plot_results:
          if optimizer.require_adj:
            plot(system,
                 data={'x': opt_x, 'u': u, 'adj': adj},
                 labels={'x': 'Resulting state trajectory',
                         'u': 'Controls from solver',
                         'adj': 'Adjoint from solver'},
                 title=title,
                 save_as=save_as)
          else:
            plot(system,
                 data={'x': opt_x, 'u': u},
                 labels={'x': 'Resulting state trajectory',
                         'u': 'Controls from solver'},
                 title=title,
                 save_as=save_as)

# TODO: why does it not work to also iterate through nlpsolvers?

if __name__=='__main__':
  unittest.main()
