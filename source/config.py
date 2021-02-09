from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
  CARTPOLE = "CARTPOLE"
  VANDERPOL = "VANDERPOL"
  SEIR = "SEIR"
  TUMOUR = "TUMOUR"
  SIMPLECASE = "SIMPLECASE"
  MOULDFUNGICIDE = "MOULDFUNGICIDE"
  BACTERIA = "BACTERIA"
  SIMPLECASEWITHBOUNDS = "SIMPLECASEWITHBOUNDS"
  CANCER = "CANCER"
  FISHHARVEST = "FISHHARVEST"
  EPIDEMICSEIRN = "EPIDEMICSEIRN"
  HIVTREATMENT = "HIVTREATMENT"
  BEARPOPULATIONS = "BEARPOPULATIONS"
  GLUCOSE = "GLUCOSE"
  TIMBERHARVEST = "TIMBERHARVEST"
  BIOREACTOR = "BIOREACTOR"
  PREDATORPREY = "PREDATORPREY"
  INVASIVEPLANT = "INVASIVEPLANT"


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
  system: SystemType = SystemType.TUMOUR
  optimizer: OptimizerType = OptimizerType.SHOOTING
  nlpsolver: NLPSolverType = NLPSolverType.IPOPT
  order: IntegrationOrder = IntegrationOrder.LINEAR
  max_iter: int = 1000            # maxiter for NLP solver
  intervals: int = 50             # used by COLLOCATION and SHOOTING
  controls_per_interval: int = 3  # used by SHOOTING
  fbsm_intervals: int = 1000      # used by FBSM

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
