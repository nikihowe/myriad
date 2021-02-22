import random
import unittest

import jax
import numpy as np

from source.config import Config, SystemType, HParams, OptimizerType, IntegrationOrder
from source.optimizers import get_optimizer
from source.systems import IndirectFHCS
from source.plotting import plot_result

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def setUp(self):
    jax.config.update("jax_enable_x64", True)

  def test_smoke(self):
    for system in SystemType:
      for optimizer in OptimizerType:
        # FBSM doesn't work on environments without adjoint dynamics
        # NOTE: do we want to add adjoint dynamics to those systems?
        if optimizer == OptimizerType.FBSM and not issubclass(system.value, IndirectFHCS):
          continue

        # TODO: currently, shooting doesn't work with:
        # predatorprey, biorector, timberharvest, fishharvest
        # This should be fixed

        with self.subTest(system=system, optimizer=optimizer):
          hp = HParams(system=system, optimizer=optimizer,
                       order=IntegrationOrder.LINEAR, intervals=20,
                       max_iter=100)
          cfg = Config(verbose=True, plot_results=True)
          random.seed(hp.seed)
          np.random.seed(hp.seed)
          _system = hp.system()
          optimizer = get_optimizer(hp, cfg, _system)
          print("calling optimizer", optimizer)
          results = optimizer.solve()
          print("solution", results[0].shape)
          print("now for plotting")

          # Plot the solution, using system-specific plotting where present
          plot_solution = getattr(_system, "plot_solution", None)
          if callable(plot_solution):
            print("using custom plotting")
            plot_solution(*results)
          else:
            print("using default plotting")
            plot_result(results, hp, save_as=hp.system.name+hp.optimizer.name+"_test")


if __name__=='__main__':
  unittest.main()

