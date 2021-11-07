# (c) 2021 Nikolaus Howe
from enum import Enum
from typing import Union

from .base import FiniteHorizonControlSystem, IndirectFHCS
from .cartpole import CartPole
from .mountain_car import MountainCar
from .pendulum import Pendulum
from .rocket_landing import RocketLanding
from .seir import SEIR
from .tumour import Tumour
from .van_der_pol import VanDerPol
from .lenhart.bacteria import Bacteria
from .lenhart.bear_populations import BearPopulations
from .lenhart.bioreactor import Bioreactor
from .lenhart.cancer_treatment import CancerTreatment
from .lenhart.epidemic_seirn import EpidemicSEIRN
from .lenhart.harvest import Harvest
from .lenhart.glucose import Glucose
from .lenhart.hiv_treatment import HIVTreatment
from .lenhart.invasive_plant import InvasivePlant
from .lenhart.mould_fungicide import MouldFungicide
from .lenhart.predator_prey import PredatorPrey
from .lenhart.simple_case import SimpleCase
from .lenhart.simple_case_with_bounds import SimpleCaseWithBounds
from .lenhart.timber_harvest import TimberHarvest


class SystemType(Enum):
  CARTPOLE = CartPole
  VANDERPOL = VanDerPol
  SEIR = SEIR
  TUMOUR = Tumour
  MOUNTAINCAR = MountainCar
  PENDULUM = Pendulum
  SIMPLECASE = SimpleCase
  MOULDFUNGICIDE = MouldFungicide
  BACTERIA = Bacteria
  SIMPLECASEWITHBOUNDS = SimpleCaseWithBounds
  CANCERTREATMENT = CancerTreatment
  EPIDEMICSEIRN = EpidemicSEIRN
  HARVEST = Harvest
  HIVTREATMENT = HIVTreatment
  BEARPOPULATIONS = BearPopulations
  GLUCOSE = Glucose
  TIMBERHARVEST = TimberHarvest
  BIOREACTOR = Bioreactor
  PREDATORPREY = PredatorPrey
  INVASIVEPLANT = InvasivePlant
  ROCKETLANDING = RocketLanding

  def __call__(self, *args, **kwargs) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
      return self.value(*args, **kwargs)


state_descriptions = {
  SystemType.CARTPOLE: [[0, 1, 2, 3], ["Position", "Angle", "Velocity", "Angular velocity"]],
  SystemType.SEIR: [[0, 1, 2, 3], ["S", "E", "I", "N"]],
  SystemType.TUMOUR: [[0, 1, 2], ["p", "q", "y"]],
  SystemType.VANDERPOL: [[0, 1], ["x0", "x1"]],
  SystemType.BACTERIA: [[0], ["Bacteria concentration"]],
  SystemType.BEARPOPULATIONS: [[0, 1, 2], ["Park population", "Forest population", "Urban population"]],
  SystemType.BIOREACTOR: [[0], ["Bacteria concentration"]],
  SystemType.CANCERTREATMENT: [[0], ["Normalized tumour density"]],
  SystemType.EPIDEMICSEIRN: [[2], ["Susceptible population", "Exposed population",
                                   "Infectious population", "Total population"]],
  SystemType.HARVEST: [[0], ["Population mass"]],
  SystemType.GLUCOSE: [[0, 1], ["Blood glucose", "Net hormonal concentration"]],
  SystemType.HIVTREATMENT: [[0], ["Healthy cells", "Infected cells", "Viral charge"]],
  SystemType.INVASIVEPLANT: [[0, 1, 2, 3, 4], ["Focus 1", "Focus 2", "Focus 3", "Focus 4", "Focus 5"]],
  SystemType.MOULDFUNGICIDE: [[0], ["Mould population"]],
  SystemType.MOUNTAINCAR: [[0, 1], ["Position", "Velocity"]],
  SystemType.PENDULUM: [[0, 1], ["Angle", "Angular velocity"]],
  SystemType.PREDATORPREY: [[0, 1], ["Predator population", "Prey population"]],
  SystemType.ROCKETLANDING: [[0, 1, 2, 3, 4, 5], ["x", "dot x", "y", "dot y", "theta", "dot theta"]],
  SystemType.SIMPLECASE: [[0], ["State"]],
  SystemType.SIMPLECASEWITHBOUNDS: [[0], ["State"]],
  SystemType.TIMBERHARVEST: [[0], ["Cumulative timber harvested"]]
}

# NOTE: the control descriptions are currently not used for plotting
control_descriptions = {
  SystemType.CARTPOLE: [[0], ["Force"]],
  SystemType.SEIR: [[0], ["Response intensity"]],
  SystemType.TUMOUR: [[0], ["Drug strength"]],
  SystemType.VANDERPOL: [[0], ["Control"]],
  SystemType.BACTERIA: [[0], ["Amount of chemical nutrient"]],
  SystemType.BEARPOPULATIONS: [[0, 1], ["Harvesting rate in park", "Harvesting rate in forest"]],
  SystemType.BIOREACTOR: [[0], ["Amount of chemical nutrient"]],
  SystemType.CANCERTREATMENT: [[0], ["Drug strength"]],
  SystemType.EPIDEMICSEIRN: [[0], ["Vaccination rate"]],
  SystemType.HARVEST: [[0], ["Harvest rate"]],
  SystemType.GLUCOSE: [[0], ["Insulin level"]],
  SystemType.HIVTREATMENT: [[0], ["Drug intensity"]],
  SystemType.MOULDFUNGICIDE: [[0], ["Fungicide level"]],
  SystemType.MOUNTAINCAR: [[0], ["Force"]],
  SystemType.PENDULUM: [[0], ["Force"]],
  SystemType.PREDATORPREY: [[0], ["Pesticide level"]],
  SystemType.ROCKETLANDING: [[0, 1], ["Thrust percent", "Angle"]],
  SystemType.SIMPLECASE: [[0], ["Control"]],
  SystemType.SIMPLECASEWITHBOUNDS: [[0], ["Control"]],
  SystemType.TIMBERHARVEST: [[0], ["Reinvestment level"]]
}


def get_name(hp):
  if hp.system == SystemType.CANCERTREATMENT:
    title = "Cancer Treatment"
  elif hp.system == SystemType.MOUNTAINCAR:
    title = 'Mountain Car'
  elif hp.system == SystemType.MOULDFUNGICIDE:
    title = 'Mould Fungicide'
  elif hp.system == SystemType.VANDERPOL:
    title = 'Van Der Pol'
  elif hp.system == SystemType.PREDATORPREY:
    title = 'Predator Prey'
  elif hp.system == SystemType.PENDULUM:
    title = "Pendulum"
  else:
    title = None
  return title
