import random
from source.systems import get_system
import unittest

import jax
import numpy as onp

from source.config import Config, SystemType, HParams, OptimizerType
from source.optimizers import get_optimizer


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def test_smoke(self):
    jax.config.update("jax_enable_x64", True)
    for system in SystemType:
      for optimizer in OptimizerType:
        with self.subTest(system=system.name, optimizer=optimizer.name):
          hp = HParams(system=system, optimizer=optimizer, ipopt_max_iter=100)
          cfg = Config(verbose=True, plot_results=True)
          random.seed(hp.seed)
          onp.random.seed(hp.seed)
          _system = get_system(hp)
          optimizer = get_optimizer(hp, cfg, _system)
          x, u = optimizer.solve()
          _system.plot_solution(x, u)


if __name__=='__main__':
  unittest.main()
