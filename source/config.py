from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
  CARTPOLE="CARTPOLE"
  VANDERPOL="VANDERPOL"
  SEIR="SEIR"
  TUMOUR="TUMOUR"
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
  COLLOCATION="COLLOCATION"
  SHOOTING="SHOOTING"
  FBSM="FBSM"


class NLPSolverType(Enum):
  # SCIPY="SCIPY"
  IPOPT="IPOPT"
  # INEXACTNEWTON="INEXACTNEWTON"
  EXTRAGRADIENT="EXTRAGRADIENT"


class IntegrationOrder(Enum):
  CONSTANT="CONSTANT"
  LINEAR="LINEAR"
  QUADRATIC="QUADRATIC"

# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=True)
class HParams:
  seed: int = 2020
  system: SystemType = SystemType.CARTPOLE
  optimizer: OptimizerType = OptimizerType.SHOOTING
  nlpsolver: NLPSolverType = NLPSolverType.IPOPT
  order: IntegrationOrder = IntegrationOrder.LINEAR
  # system: SystemType = SystemType.FISHHARVEST
  # optimizer: OptimizerType = OptimizerType.FBSM
  # Solver
  ipopt_max_iter: int = 1_000
  # Trajectory Optimizer
  intervals: int = 30 # collocation and shooting 
  # TODO: make it include the single shooting case of 1 interval. Right now that breaks
  controls_per_interval: int = 1 # multiple shooting

  #Indirect method optimizer
  steps: int = 1000


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config():
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True