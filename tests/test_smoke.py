import random
import unittest

import jax
import numpy as np

from source.config import Config, SystemType, HParams, OptimizerType, NLPSolverType, IntegrationOrder
from source.optimizers import get_optimizer
from source.systems import get_system


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def test_smoke(self):
    jax.config.update("jax_enable_x64", True)
    for system in SystemType:
      for optimizer in OptimizerType:
        with self.subTest(system=system, optimizer=optimizer):
          hp = HParams(system=system, optimizer=optimizer, order=IntegrationOrder.CONSTANT)
          cfg = Config(verbose=True, plot_results=True)
          random.seed(hp.seed)
          np.random.seed(hp.seed)
          _system = get_system(hp)
          optimizer = get_optimizer(hp, cfg, _system)
          print("calling optimizer", optimizer)
          results = optimizer.solve()
          print("solution", results[0].shape)
          _system.plot_solution(*results)
# TODO: why does it not work to also iterate through nlpsolvers?

if __name__=='__main__':
  unittest.main()
