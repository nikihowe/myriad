import random
import unittest

import numpy as onp

from source.config import Config, DynamicsType, HParams
from source.experiment import experiment


# Test that experiments run without raising exceptions
class SmokeTest(unittest.TestCase):
  def test_smoke(self):
    for dynamics in DynamicsType:
      with self.subTest(dynamics=dynamics.name):
        hp = HParams(dynamics=dynamics, slsqp_maxiter=100)
        cfg = Config(verbose=False, plot_results=False)
        random.seed(hp.seed)
        onp.random.seed(hp.seed)
        experiment(hp, cfg)


if __name__=='__main__':
  unittest.main()
