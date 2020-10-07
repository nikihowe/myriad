from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
  CARTPOLE="CARTPOLE"
  VANDERPOL="VANDERPOL"
  SEIR="SEIR"
  TUMOUR="TUMOUR"
  LENHART1="LENHART1"
  LENHART5 = "LENHART5"


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
