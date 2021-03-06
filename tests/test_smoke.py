import random
import unittest

import jax
import numpy as np

from myriad.config import Config, SystemType, HParams, OptimizerType
from myriad.trajectory_optimizers import get_optimizer
from myriad.systems import IndirectFHCS
from myriad.plotting import plot_result

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

approaches = {
  'fbsm': {'optimizer': OptimizerType.FBSM, 'fbsm_intervals': 1000},
  'single_shooting': {'optimizer': OptimizerType.SHOOTING, 'intervals': 1, 'controls_per_interval': 90},
  'multiple_shooting_3_controls': {'optimizer': OptimizerType.SHOOTING, 'intervals': 30, 'controls_per_interval': 3},
  'multiple_shooting_1_control': {'optimizer': OptimizerType.SHOOTING, 'intervals': 90, 'controls_per_interval': 1},
  'collocation': {'optimizer': OptimizerType.COLLOCATION, 'intervals': 90, 'controls_per_interval': 1}
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

        # Invasive plant is a discrete system, so it only works with FBSM
        if hp.system == SystemType.INVASIVEPLANT:
          continue

        with self.subTest(system=hp.system, optimizer=hp.optimizer):
          cfg = Config(verbose=True)
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

