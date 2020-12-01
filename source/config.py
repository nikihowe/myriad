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


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=True)
class HParams:
  seed: int = 2020
  system: SystemType = SystemType.BIOREACTOR
  optimizer: OptimizerType = OptimizerType.FBSM
  # Solver
  ipopt_max_iter: int = 3000
  # Trajectory Optimizer
  intervals: int = 15 # collocation and shooting
  controls_per_interval: int = 5 # multiple shooting

  #Indirect method optimizer
  steps: int = 1000


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config():
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True