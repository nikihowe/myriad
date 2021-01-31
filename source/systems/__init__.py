from source.config import HParams, SystemType
from typing import Union

from .base import FiniteHorizonControlSystem
from .base import IndirectFHCS
from .cartpole import CartPole
from .vanderpol import VanDerPol
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


def get_system(hp: HParams) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
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
  elif hp.system == SystemType.MOLDFUNGICIDE:
    return MoldFungicide()
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
