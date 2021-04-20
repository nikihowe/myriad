from dataclasses import dataclass
from enum import Enum

from myriad.systems import SystemType


class OptimizerType(Enum):
  """Parser argument. Optimizing strategy used to solve the OCP"""
  COLLOCATION = "COLLOCATION"
  """Collocation method, select with `--optimizer=COLLOCATION`"""
  SHOOTING = "SHOOTING"
  """Shooting method (single and multi), select with `--optimizer=SHOOTING`"""
  FBSM = "FBSM"
  """Forward-Backward Sweep method (indirect), select with `--optimizer=FBSM`"""


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
  """The hyperparameters of the experiment. Modifying those should slightly change the results"""
  # seed: int = 2020
  # """The seed of the RNG; set with `seed=<int>`"""
  system: SystemType = SystemType.EPIDEMICSEIRN
  """The system to run the experiment on; set with `system=<SystemType>`. Value can be any of [SystemType](https://simonduflab.github.io/optimal-control/html/myriad/systems/index.html#myriad.systems.SystemType)"""
  optimizer: OptimizerType = OptimizerType.SHOOTING
  """The optimizer to solve the OCP. Select from [OptimizerType](https://simonduflab.github.io/optimal-control/html/myriad/config.html#myriad.config.OptimizerType)"""
  nlpsolver: NLPSolverType = NLPSolverType.SLSQP
  """The NLP solver to use during optimization; set with `nlpsolver=<NLPSolverType>`. Value can be any of [NLPSolverType](https://simonduflab.github.io/optimal-control/html/myriad/config.html#myriad.config.NLPSolverType)"""
  order: IntegrationOrder = IntegrationOrder.LINEAR
  """Order when integrating; set with `order=<IntegrationOrder>`. Value can be any of [IntegrationOrder](https://simonduflab.github.io/optimal-control/html/myriad/config.html#myriad.config.IntegrationOrder)"""
  max_iter: int = 1000              # maxiter for NLP solver
  """Maximum iteration for NLP solvers; set wit `max_iter=<int>`"""
  intervals: int = 1               # used by COLLOCATION and SHOOTING
  """Number of intervals to use for optimizer, used by `COLLOCATION` and `SHOOTING` methods; set with `intervals=<int>`"""
  controls_per_interval: int = 100   # used by SHOOTING
  """Number of controls per interval, used by `SHOOTING` (multiple) method; set with `controls_per_interval=<int>`"""
  fbsm_intervals: int = 1000        # used by FBSM # TODO: merge with previous interval option
  """Used by `FBSM` method; set with `fbsm_intervals=<int>`"""

  # Collocation requires exactly one control per interval
  def __post_init__(self):
    if self.optimizer == OptimizerType.COLLOCATION:
      self.controls_per_interval = 1


@dataclass(eq=True, frozen=True)
class Config:
  """Secondary configurations that should not change experiment results
  and should be largely used for debugging"""
  verbose: bool = True
  """Verbose mode; default to `True`"""
  jit: bool = True
  """Enable [`@jit`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions) compilation; default to `True`"""
  plot_results: bool = True
  """Plot results after running the experiment; default to `True`"""
