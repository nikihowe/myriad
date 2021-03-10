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
  seed: int = 2020
  system: SystemType = SystemType.EPIDEMICSEIRN
  optimizer: OptimizerType = OptimizerType.FBSM
  nlpsolver: NLPSolverType = NLPSolverType.SLSQP
  order: IntegrationOrder = IntegrationOrder.LINEAR
  max_iter: int = 1000              # maxiter for NLP solver
  intervals: int = 1                # used by COLLOCATION and SHOOTING
  controls_per_interval: int = 30   # used by SHOOTING
  fbsm_intervals: int = 1000        # used by FBSM

  # Collocation requires exactly one control per interval
  def __post_init__(self):
    if self.optimizer == OptimizerType.COLLOCATION:
      self.controls_per_interval = 1


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config:
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True
