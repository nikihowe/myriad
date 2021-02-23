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

approaches = {
  'fbsm': {'optimizer': OptimizerType.FBSM, 'fbsm_intervals': 1000},
  'single_shooting': {'optimizer': OptimizerType.SHOOTING, 'intervals': 1, 'controls_per_interval': 60},
  'multiple_shooting': {'optimizer': OptimizerType.SHOOTING, 'intervals': 20, 'controls_per_interval': 3},
  'collocation': {'optimizer': OptimizerType.COLLOCATION, 'intervals': 60, 'controls_per_interval': 1}
}


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def setUp(self):
    jax.config.update("jax_enable_x64", True)

  def test_smoke(self):
    for system in SystemType:
      for approach in approaches:
        hp = HParams(system=system, **approaches[approach])  # unpack the hparams for this approach

        # TODO: add adjoint dynamics to those systems, so that FBSM can be used
        # (FBSM doesn't work on environments without adjoint dynamics)
        if hp.optimizer == OptimizerType.FBSM and not issubclass(system.value, IndirectFHCS):
          continue

        # TODO: make shooting work with these (probably requires small environment modification)
        if hp.optimizer == OptimizerType.SHOOTING and hp.system in [SystemType.PREDATORPREY,
                                                                    SystemType.BIOREACTOR,
                                                                    SystemType.TIMBERHARVEST,
                                                                    SystemType.FISHHARVEST]:
          continue

        # Invasive plant is a distrete system, so it only works with FBSM
        if hp.system == SystemType.INVASIVEPLANT:
          continue

        with self.subTest(system=hp.system, optimizer=hp.optimizer):
          cfg = Config(verbose=True, plot_results=True)
          random.seed(hp.seed)
          np.random.seed(hp.seed)
          _system = hp.system()
          optimizer = get_optimizer(hp, cfg, _system)
          print("calling optimizer", hp.optimizer.name)
          results = optimizer.solve()
          print("solution", results[0].shape)
          print("now for plotting")

          # Plot the solution, using system-specific plotting where present
          # plot_solution = getattr(_system, "plot_solution", None)
          # if callable(plot_solution):
          #   print("using custom plotting")
          #   plot_solution(*results)
          # else:
          print("using default plotting")
          plot_result(results, hp, save_as=approach+hp.system.name+"_test")


if __name__=='__main__':
  unittest.main()

