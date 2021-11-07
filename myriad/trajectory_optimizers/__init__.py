# (c) 2021 Nikolaus Howe
from typing import Union

from myriad.trajectory_optimizers.base import TrajectoryOptimizer, IndirectMethodOptimizer
from myriad.trajectory_optimizers.collocation.trapezoidal import TrapezoidalCollocationOptimizer
from myriad.trajectory_optimizers.collocation.hermite_simpson import HermiteSimpsonCollocationOptimizer
from myriad.trajectory_optimizers.forward_backward_sweep import FBSM
from myriad.trajectory_optimizers.shooting import MultipleShootingOptimizer
from myriad.config import Config, HParams, QuadratureRule, OptimizerType


def get_optimizer(hp: HParams, cfg: Config, system#: Union[FiniteHorizonControlSystem, IndirectFHCS]
                  ) -> Union[TrajectoryOptimizer, IndirectMethodOptimizer]:
  """ Helper function to fetch the desired optimizer for system resolution"""
  if hp.optimizer == OptimizerType.COLLOCATION:
    if hp.quadrature_rule == QuadratureRule.TRAPEZOIDAL:
      optimizer = TrapezoidalCollocationOptimizer(hp, cfg, system)
    elif hp.quadrature_rule == QuadratureRule.HERMITE_SIMPSON:
      optimizer = HermiteSimpsonCollocationOptimizer(hp, cfg, system)
    else:
      raise KeyError
  elif hp.optimizer == OptimizerType.SHOOTING:
    optimizer = MultipleShootingOptimizer(hp, cfg, system)
  elif hp.optimizer == OptimizerType.FBSM:
    optimizer = FBSM(hp, cfg, system)
  else:
    raise KeyError
  return optimizer

  # def __call__(self, *args, **kwargs) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
  #     return self.value(*args, **kwargs)
