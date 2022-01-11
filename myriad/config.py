# (c) 2021 Nikolaus Howe
from typing import Tuple

import jax

from dataclasses import dataclass
from enum import Enum

from myriad.systems import SystemType


class OptimizerType(Enum):
  """Parser argument. Optimizing strategy used to solve the OCP"""
  # _settings_ = NoAlias
  COLLOCATION = "COLLOCATION"
  SHOOTING = "SHOOTING"
  FBSM = "FBSM"


class SamplingApproach(Enum):
  UNIFORM = 'UNIFORM'
  TRUE_OPTIMAL = 'TRUE_OPTIMAL'
  RANDOM_WALK = 'RANDOM_WALK'
  CURRENT_OPTIMAL = 'CURRENT_OPTIMAL'  # TODO: current optimal is broken at the moment, because we're not
                                       # TODO: the guess around which we are sampling
  # RANDOM_GRID = 'RANDOM_GRID'
  # This ^ isn't implemented yet. It's unclear how helpful it would be

  # FULL_GRID = 'FULL_GRID'
  # We're not doing the FULL GRID anymore because it breaks the idea of generating trajectories.
  # But it would be interesting to compare performance against, since in some sense this is the
  # theoretical best. I wonder how resilient it would be to noise though.
  # ENDTOEND = "ENDTOEND"

  # ORNSTECK_BLABLA = "snnth"
  # Another one we should try to implement


class NLPSolverType(Enum):
  SLSQP = "SLSQP"  # Scipy's SLSQP
  TRUST = "TRUST"  # Scipy's trust-constr
  IPOPT = "IPOPT"  # ipopt
  # INEXACTNEWTON="INEXACTNEWTON"
  EXTRAGRADIENT = "EXTRAGRADIENT"  # an extragradient-based solver


class IntegrationMethod(Enum):
  EULER = "CONSTANT"
  HEUN = "LINEAR"
  MIDPOINT = "MIDPOINT"
  RK4 = "RK4"


class QuadratureRule(Enum):
  TRAPEZOIDAL = "TRAPEZOIDAL"
  HERMITE_SIMPSON = "HERMITE_SIMPSON"


# Hyperparameters which change experiment results
@dataclass(eq=True, frozen=False)  # or frozen == False
class HParams:
  """The hyperparameters of the experiment. Modifying those should change the results"""
  seed: int = 2019
  system: SystemType = SystemType.CANCERTREATMENT
  optimizer: OptimizerType = OptimizerType.SHOOTING
  nlpsolver: NLPSolverType = NLPSolverType.EXTRAGRADIENT
  integration_method: IntegrationMethod = IntegrationMethod.HEUN
  quadrature_rule: QuadratureRule = QuadratureRule.TRAPEZOIDAL

  max_iter: int = 1000  # maxiter for NLP solver (usually 1000)
  intervals: int = 1  # used by COLLOCATION and SHOOTING
  controls_per_interval: int = 100  # used by SHOOTING
  fbsm_intervals: int = 1000  # used by FBSM

  sampling_approach: SamplingApproach = SamplingApproach.RANDOM_WALK
  train_size: int = 10
  val_size: int = 3
  test_size: int = 3
  sample_spread: float = 0.05
  start_spread: float = 0.1
  noise_level: float = 0.01 * 0.
  to_smooth: bool = False
  learning_rate: float = 0.001
  minibatch_size: int = 16
  num_epochs: int = 1_000_001
  num_experiments: int = 5
  loss_recording_frequency: int = 1000
  plot_progress_frequency: int = 10_000
  early_stop_threshold: int = 30_000  # 70 for cartpole, 1 for cancertreatment
  early_stop_check_frequency: int = 1000
  hidden_layers: Tuple[int] = (100, 100)
  num_unrolled: int = 5
  eta_x: float = 1e-1
  eta_lmbda: float = 1e-3
  adam_lr: float = 1e-4

  def __post_init__(self):
    if self.optimizer == OptimizerType.COLLOCATION:
      self.controls_per_interval = 1
    if self.nlpsolver == NLPSolverType.EXTRAGRADIENT:
      self.max_iter *= 10

    # For convenience, record number of steps and stepsize
    system = self.system()
    self.num_steps = self.intervals * self.controls_per_interval
    self.stepsize = system.T / self.num_steps
    self.key = jax.random.PRNGKey(self.seed)
    self.state_size = system.x_0.shape[0]
    self.control_size = system.bounds.shape[0] - self.state_size

    # Fix the minibatch size if we're working with small datasets
    self.minibatch_size = min([self.minibatch_size, self.train_size, self.val_size, self.test_size])


@dataclass(eq=True, frozen=False)
class Config:
  """Secondary configurations that should not change experiment results
  and should be largely used for debugging"""
  verbose: bool = True
  """Verbose mode; default to `True`"""
  jit: bool = True
  """Enable [`@jit`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions) compilation; default to `True`"""
  plot: bool = True
  """Plot progress during (and results after) the experiment; default to `True`"""
  pretty_plotting: bool = True
  """Only plot the true trajectory, ignoring the solver state output"""
  load_params_if_saved: bool = True
  figsize: Tuple[float, float] = (8, 6)
  file_extension: str = 'png'  # pdf, pgf, png
