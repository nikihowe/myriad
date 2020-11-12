from dataclasses import dataclass
from enum import Enum
from typing import  Optional


class SystemType(Enum):
  CARTPOLE="CARTPOLE"
  VANDERPOL="VANDERPOL"
  SEIR="SEIR"
  TUMOUR="TUMOUR"
  SIMPLECASE = "SIMPLECASE"
  MOLDFUNGICIDE = "MOLDFUNGICIDE"
  BACTERIA = "BACTERIA"
  LENHART4 = "LENHART4"
  LENHART5 = "LENHART5"
  LENHART6 = "LENHART6"
  LENHART7 = "LENHART7"
  LENHART8 = "LENHART8"
  LENHART9 = "LENHART9"
  LENHART10 = "LENHART10"
  LENHART11 = "LENHART11"
  LENHART12 = "LENHART12"
  LENHART13 = "LENHART13"
  LENHART14 = "LENHART14"


class OptimizerType(Enum):
  COLLOCATION="COLLOCATION"
  SHOOTING="SHOOTING"
  FBSM="FBSM"


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=True)
class HParams:
  seed: int = 2020
  system: SystemType = SystemType.CARTPOLE
  optimizer: OptimizerType = OptimizerType.COLLOCATION
  # Solver
  ipopt_max_iter: int = 3000
  # Trajectory Optimizer
  intervals: int = 25 # collocation and shooting
  controls_per_interval: int = 5 # multiple shooting

  #Indirect method optimizer
  steps: int = 1000


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config():
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True