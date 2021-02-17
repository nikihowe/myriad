import random
import unittest

import jax
import numpy as np

from source.config import Config, SystemType, HParams, OptimizerType, IntegrationOrder
from source.optimizers import get_optimizer
from source.systems import IndirectFHCS


class SmokeTest(unittest.TestCase):
  def setUp(self):
    jax.config.update("jax_enable_x64", True)

  def test_smoke(self):
    for system in SystemType:
      for optimizer in OptimizerType:
        if optimizer == OptimizerType.FBSM and not issubclass(system.value, IndirectFHCS):
          continue
        with self.subTest(system=system, optimizer=optimizer):
          hp = HParams(system=system, optimizer=optimizer, order=IntegrationOrder.LINEAR, intervals=20, max_iter=100)
          cfg = Config(verbose=True, plot_results=True)
          random.seed(hp.seed)
          np.random.seed(hp.seed)
          _system = hp.system()
          optimizer = get_optimizer(hp, cfg, _system)
          print("calling optimizer", optimizer)
          results = optimizer.solve()
          print("solution", results[0].shape)
          _system.plot_solution(*results)

if __name__=='__main__':
  unittest.main()
