from dataclasses import dataclass
from enum import Enum
from typing import  Optional


class SystemType(Enum):
  CARTPOLE="CARTPOLE"
  VANDERPOL="VANDERPOL"
  SEIR="SEIR"
  TUMOUR="TUMOUR"
  LENHART1="LENHART1"
  LENHART5 = "LENHART5"
  LENHART6 = "LENHART6"
  LENHART7 = "LENHART7"
  LENHART8 = "LENHART8"
  LENHART9 = "LENHART9"
  LENHART10 = "LENHART10"


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

# Third configurations, that manipulate directly the dynamics and modify the environment
@dataclass(eq=True, frozen=True)
class MParams:                       # TODO: this is disgusting, let's reorganize...
  x_0 : Optional[float] = None
  x_T : Optional[float] = None
  T : Optional[float] = None
  A: Optional[float] = None
  B: Optional[float] = None
  C: Optional[float] = None
  D: Optional[float] = None
  E: Optional[float] = None
  F: Optional[float] = None
  G: Optional[float] = None
  H: Optional[float] = None
  I: Optional[float] = None
  M: Optional[float] = None

