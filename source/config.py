from dataclasses import dataclass
from enum import Enum


class SystemType(Enum):
  CARTPOLE="CARTPOLE"
  VANDERPOL="VANDERPOL"
  SEIR="SEIR"
  TUMOUR="TUMOUR"


class OptimizerType(Enum):
  COLLOCATION="COLLOCATION"
  SHOOTING="SHOOTING"


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=True)
class HParams:
  seed: int = 2020
  system: SystemType = SystemType.CARTPOLE
  optimizer: OptimizerType = OptimizerType.COLLOCATION
  # Solver
  slsqp_maxiter: int = 1000
  slsqp_ftol: float = 1e-6
  # Trajectory Optimizer
  intervals: int = 25 # collocation and shooting
  controls_per_interval: int = 5 # multiple shooting


# Secondary configurations which should not change experiment results
@dataclass(eq=True, frozen=True)
class Config():
  verbose: bool = True
  jit: bool = True
  plot_results: bool = True
