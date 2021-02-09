from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union, Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from source.config import SystemType, HParams


@dataclass
class FiniteHorizonControlSystem(object):
  _type: SystemType
  x_0: jnp.array            # Starting state
  x_T: Optional[jnp.array]  # Required final state, if any
  T: float                  # Duration of trajectory
  bounds: jnp.ndarray       # State and control bounds
  terminal_cost: bool       # Is there an additional cost based on the final state?
  discrete: bool = False    # Is our cost discrete in time?

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.array]) -> jnp.ndarray:
    raise NotImplementedError
  
  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.array]) -> float:
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: Optional[jnp.array], T: Optional[jnp.array] = None) -> float:
    return 0

  # def plot_solution(self, data: Dict[str, jnp.ndarray],
  #                   labels: Dict[str, str], title: Optional[str] = None,
  #                   save_as: Optional[str] = None) -> None:
  #   raise NotImplementedError


@dataclass
class IndirectFHCS(FiniteHorizonControlSystem, ABC):
  adj_T: Optional[jnp.array] = None  # adjoint at time T
  guess_a: Optional[float] = None  # Initial guess for secant method
  guess_b: Optional[float] = None  # Initial guess for secant method

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
  #   raise NotImplementedError


def get_system(hp: HParams) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
  from .BaseSystems.CartPole import CartPole
  from .BaseSystems.VanDerPol import VanDerPol
  from .BaseSystems.SEIR import SEIR
  from .BaseSystems.Tumour import Tumour
  from .LenhartSystems.SimpleCase import SimpleCase  # TODO: This is just dirty : to clean
  from .LenhartSystems.MouldFungicide import MouldFungicide
  from .LenhartSystems.Bacteria import Bacteria
  from .LenhartSystems.SimpleCaseWithBounds import SimpleCaseWithBounds
  from .LenhartSystems.Cancer import Cancer
  from .LenhartSystems.FishHarvest import FishHarvest
  from .LenhartSystems.EpidemicSEIRN import EpidemicSEIRN
  from .LenhartSystems.HIVTreatment import HIVTreatment
  from .LenhartSystems.BearPopulations import BearPopulations
  from .LenhartSystems.Glucose import Glucose
  from .LenhartSystems.TimberHarvest import TimberHarvest
  from .LenhartSystems.Bioreactor import Bioreactor
  from .LenhartSystems.PredatorPrey import PredatorPrey
  from .LenhartSystems.InvasivePlant import InvasivePlant
  
  if hp.system == SystemType.CARTPOLE:
    return CartPole()
  elif hp.system == SystemType.VANDERPOL:
    return VanDerPol()
  elif hp.system == SystemType.SEIR:
    return SEIR()
  elif hp.system == SystemType.TUMOUR:
    return Tumour()
  elif hp.system == SystemType.SIMPLECASE:
    return SimpleCase()
  elif hp.system == SystemType.MOULDFUNGICIDE:
    return MouldFungicide()
  elif hp.system == SystemType.BACTERIA:
    return Bacteria()
  elif hp.system == SystemType.SIMPLECASEWITHBOUNDS:
    return SimpleCaseWithBounds()
  elif hp.system == SystemType.CANCER:
    return Cancer()
  elif hp.system == SystemType.FISHHARVEST:
    return FishHarvest()
  elif hp.system == SystemType.EPIDEMICSEIRN:
    return EpidemicSEIRN()
  elif hp.system == SystemType.HIVTREATMENT:
    return HIVTreatment()
  elif hp.system == SystemType.BEARPOPULATIONS:
    return BearPopulations()
  elif hp.system == SystemType.GLUCOSE:
    return Glucose()
  elif hp.system == SystemType.TIMBERHARVEST:
    return TimberHarvest()
  elif hp.system == SystemType.BIOREACTOR:
    return Bioreactor()
  elif hp.system == SystemType.PREDATORPREY:
    return PredatorPrey()
  elif hp.system == SystemType.INVASIVEPLANT:
    return InvasivePlant()
  else:
    raise KeyError

