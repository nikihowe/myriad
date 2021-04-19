from dataclasses import dataclass
from enum import Enum

from source.systems import SystemType


class OptimizerType(Enum):
  COLLOCATION = "COLLOCATION"
  SHOOTING = "SHOOTING"
  FBSM = "FBSM"


class NLPSolverType(Enum):
  SLSQP = "SLSQP"  # Scipy's SLSQP
  TRUST = "TRUST"  # Scipy's trust-constr
  IPOPT = "IPOPT"  # ipopt
  # INEXACTNEWTON="INEXACTNEWTON"
  EXTRAGRADIENT = "EXTRAGRADIENT"  # an extragradient-based solver


class IntegrationOrder(Enum):
  CONSTANT = "CONSTANT"
  LINEAR = "LINEAR"
  QUADRATIC = "QUADRATIC"


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=False)
class HParams:
  """
  Hyperparameters

  :param system: System on which to perform trajectory optimization
  :param optimizer: Optimizer to use for trajectory optimization
  :param nlpsolver: Solver to use on nonlinear program created by optimizer
  :param order: Order of interpolation to use for optimizer
  :param max_iter: Iteration limit for nlpsolver
  :param intervals: Number of intervals to use for optimizer (applicable only to direct trajectory optimizers)
  :param controls_per_interval: Number of controls per interval (applicable only to multiple shooting)
  :param fbsm_intervals: Number of intervals for optimizer (applicable only to indirect trajectory optimizers)
  """
  # seed: int = 2020
  system: SystemType = SystemType.EPIDEMICSEIRN
  optimizer: OptimizerType = OptimizerType.SHOOTING
  nlpsolver: NLPSolverType = NLPSolverType.SLSQP
  order: IntegrationOrder = IntegrationOrder.LINEAR
  max_iter: int = 1000
  intervals: int = 1
  controls_per_interval: int = 100
  fbsm_intervals: int = 1000

  # Collocation requires exactly one control per interval
  def __post_init__(self):
    if self.optimizer == OptimizerType.COLLOCATION:
      self.controls_per_interval = 1


# Secondary configurations which should not change experiment results
# and should be largely used for debugging
@dataclass(eq=True, frozen=True)
class Config:
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True
