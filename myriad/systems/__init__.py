from enum import Enum
from typing import Union

from .base import FiniteHorizonControlSystem
from .base import IndirectFHCS
from .cartpole import CartPole
from .vanderpol import VanDerPol
from .mountaincar import MountainCar
from .pendulum import Pendulum
from .seir import SEIR
from .tumour import Tumour
from .lenhart.simple_case import SimpleCase
from .lenhart.mold_fungicide import MoldFungicide
from .lenhart.bacteria import Bacteria
from .lenhart.simple_case_with_bounds import SimpleCaseWithBounds
from .lenhart.cancer import Cancer
from .lenhart.fish_harvest import FishHarvest
from .lenhart.epidemic_seirn import EpidemicSEIRN
from .lenhart.hiv_treatment import HIVTreatment
from .lenhart.bear_populations import BearPopulations
from .lenhart.glucose import Glucose
from .lenhart.timber_harvest import TimberHarvest
from .lenhart.bioreactor import Bioreactor
from .lenhart.predator_prey import PredatorPrey
from .lenhart.invasive_plant import InvasivePlant


class SystemType(Enum):
  CARTPOLE = CartPole
  VANDERPOL = VanDerPol
  SEIR = SEIR
  TUMOUR = Tumour
  MOUNTAINCAR = MountainCar
  PENDULUM = Pendulum
  SIMPLECASE = SimpleCase
  MOLDFUNGICIDE = MoldFungicide
  BACTERIA = Bacteria
  SIMPLECASEWITHBOUNDS = SimpleCaseWithBounds
  CANCER = Cancer
  FISHHARVEST = FishHarvest
  EPIDEMICSEIRN = EpidemicSEIRN
  HIVTREATMENT = HIVTreatment
  BEARPOPULATIONS = BearPopulations
  GLUCOSE = Glucose
  TIMBERHARVEST = TimberHarvest
  BIOREACTOR = Bioreactor
  PREDATORPREY = PredatorPrey
  INVASIVEPLANT = InvasivePlant


  def __call__(self, *args, **kwargs) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
      return self.value(*args, **kwargs)
