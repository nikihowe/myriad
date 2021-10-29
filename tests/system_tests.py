# (c) Nikolaus Howe 2021
import sys
import unittest

from source.config import IntegrationMethod, NLPSolverType, OptimizerType, SystemType
from source.useful_scripts import run_setup
from run import run_trajectory_opt

hp, cfg = run_setup(sys.argv, gin_path='../source/gin-configs/default.gin')


class SystemTests(unittest.TestCase):
  def test_systems(self):
    global hp, cfg

    hp.seed = 42
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    # Try to optimize each system (except for the discrete one)
    for system_type in SystemType:
      # Skip Invasive Plant, since it's not a continuous system
      if system_type == SystemType.INVASIVEPLANT:
        continue
      hp.system = system_type
      run_trajectory_opt(hp, cfg)


if __name__ == '__main__':
  unittest.main()
