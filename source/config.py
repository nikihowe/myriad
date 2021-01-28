from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
  CARTPOLE = "CARTPOLE"
  VANDERPOL = "VANDERPOL"
  SEIR = "SEIR"
  TUMOUR = "TUMOUR"
  SIMPLECASE = "SIMPLECASE"
  MOLDFUNGICIDE = "MOLDFUNGICIDE"
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
  SLSQP = "SLSQP"
  TRUST = "TRUST"
  IPOPT = "IPOPT"
  # INEXACTNEWTON="INEXACTNEWTON"
  EXTRAGRADIENT = "EXTRAGRADIENT"


class IntegrationOrder(Enum):
  CONSTANT = "CONSTANT"
  LINEAR = "LINEAR"
  QUADRATIC = "QUADRATIC"


class SamplingApproach(Enum):
  FIXED = "FIXED"
  FIXED_OPTIMAL = 'FIXED_OPTIMAL'
  PLANNING = "PLANNING"
  ENDTOEND = "ENDTOEND"


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=False)
class HParams:
  seed: int = 2020
  system: SystemType = SystemType.CARTPOLE
  optimizer: OptimizerType = OptimizerType.SHOOTING
  nlpsolver: NLPSolverType = NLPSolverType.IPOPT
  order: IntegrationOrder = IntegrationOrder.LINEAR
  sampling_approach: SamplingApproach = SamplingApproach.PLANNING
  # Solver
  max_iter: int = 1000
  # Trajectory Optimizer
  intervals: int = 20  # collocation and shooting
  controls_per_interval: int = 3  # multiple shooting

  # Indirect method optimizer
  steps: int = 100

  def __post_init__(self):
    if self.optimizer == OptimizerType.COLLOCATION:
      self.controls_per_interval = 1


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config:
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True
