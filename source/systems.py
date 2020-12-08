from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import SystemType, HParams


@dataclass
class FiniteHorizonControlSystem(object):
  _type: SystemType
  x_0: jnp.array  # state at time 0
  x_T: Optional[jnp.array]  # state at time T
  T: float  # duration of trajectory
  bounds: jnp.ndarray  # State and control bounds
  terminal_cost: bool  # Whether only the final state and control are inputs to the cost
  discrete: bool = False  # Whether we are working with a system with continuous cost or not

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(jnp.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(jnp.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray]) -> jnp.ndarray:
    raise NotImplementedError
  
  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.array]) -> float:
    raise NotImplementedError

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: Optional[jnp.array], T: Optional[jnp.array] = None) -> float:
    return 0

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    raise NotImplementedError


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

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray, adj: Optional[jnp.ndarray] = None) -> None:
    raise NotImplementedError


def get_system(hp: HParams) -> Union[FiniteHorizonControlSystem, IndirectFHCS]:
  from .LenhartSystems.SimpleCase import SimpleCase  # TODO: This is just dirty : to clean
  from .LenhartSystems.MoldFungicide import MoldFungicide
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


class CartPole(FiniteHorizonControlSystem):
  def __init__(self):
    # Physical parameters for the cart-pole example (Table 3)
    self.m1 = 1.0  # kg mass of cart
    self.m2 = 0.3  # kg mass of pole
    self.l = 0.5   # m pole length
    self.g = 9.81  # m/s^2 gravity acceleration
    self.u_max = 20  # N maximum actuator force
    self.d_max = 2.0 # m extent of the rail that cart travels on
    self.d = 1.0   # m distance traveled during swing-up

    super().__init__(
      _type = SystemType.CARTPOLE,
      x_0 = jnp.array([0., 0., 0., 0.]),  # Starting state (Eq. 6.9)
      x_T = jnp.array([self.d, jnp.pi, 0., 0.]),  # Ending state (Eq. 6.9)
      T = 2.0,  # s duration of swing-up,
      bounds = jnp.array([
        [-self.d_max, self.d_max],  # Eq. 6.7
        [-2*jnp.pi, 2*jnp.pi],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-self.u_max, self.u_max],  # Control bounds (Eq. 6.8)
      ]),
      terminal_cost = False,
    )

  # Cart-Pole Example: System Dynamics (Section 6.1)
  def dynamics(self, x_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
    q1, q2, q̇1, q̇2 = x_t
    # Eq. 6.1
    q̈1 = ((self.l * self.m2 * jnp.sin(q2) * q̇2**2 + u_t + self.m2 * self.g * jnp.cos(q2) * jnp.sin(q2))
           / (self.m1 + self.m2 * (1 - jnp.cos(q2)**2)))
    q̈1 = jnp.squeeze(q̈1)
    # Eq. 6.2
    q̈2 = - ((self.l * self.m2 * jnp.cos(q2) * q̇2**2 + u_t * jnp.cos(q2) + (self.m1 + self.m2) * self.g * jnp.sin(q2))
             / (self.l * self.m1 + self.l * self.m2 * (1 - jnp.cos(q2)**2)))
    q̈2 = jnp.squeeze(q̈2)
    return jnp.array([q̇1, q̇2, q̈1, q̈2])
  
  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    # Eq. 6.3
    return u_t ** 2
  
  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    x = pd.DataFrame(x, columns=['q1', 'q2', 'q̈1', 'q̈2'])

    # Plot optimal trajectory (Figure 10)
    sns.set(style='darkgrid')
    plt.figure(figsize=(9, 6))
    ts_x = jnp.linspace(0, self.T, x.shape[0])
    ts_u = jnp.linspace(0, self.T, u.shape[0])

    plt.subplot(3, 1, 1)
    plt.ylabel('position (m)')
    plt.xlim(0, 2)
    plt.ylim(0, 1.5)
    plt.plot(ts_x, x['q1'], '-bo', clip_on=False, zorder=10)

    plt.subplot(3, 1, 2)
    plt.ylabel('angle (rad)')
    plt.plot(ts_x, x['q2'], '-bo', clip_on=False, zorder=10)
    plt.xlim(0, 2)
    plt.ylim(-2, 4)

    plt.subplot(3, 1, 3)
    plt.ylabel('force (N)')
    # plt.plot(ts_u, u, '-bo', clip_on=False, zorder=10)
    plt.step(ts_u, u, where="post", clip_on=False)
    plt.xlim(0, 2)
    plt.ylim(-20, 10)

    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()

    
class VanDerPol(FiniteHorizonControlSystem):
  def __init__(self):
    super().__init__(
      _type = SystemType.VANDERPOL,
      x_0 = jnp.array([0., 1.]),
      x_T = jnp.zeros(2),
      T = 10.0,
      bounds = jnp.array([
        [-jnp.inf, jnp.inf], # state 1
        [-jnp.inf, jnp.inf], # state 2
        [-0.75, 1.0],        # control
      ]),
      terminal_cost = False,
    )

  def dynamics(self, x_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
    x0, x1 = x_t
    _x0 = jnp.squeeze((1. - x1**2) * x0 - x1 + u_t)
    _x1 = jnp.squeeze(x0)
    return jnp.asarray([_x0, _x1])
  
  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return x_t.T @ x_t + u_t ** 2

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    x = pd.DataFrame(x, columns=['x0', 'x1'])

    sns.set(style='darkgrid')
    plt.figure(figsize=(9, 4))
    ts_u = jnp.linspace(0, self.T, u.shape[0])

    plt.subplot(1, 2, 1)
    plt.plot(x['x0'], x['x1'])
    
    plt.subplot(1, 2, 2)
    plt.step(ts_u, u, where="post")
    plt.xlabel('time (s)')

    plt.tight_layout()
    plt.show()


class SEIR(FiniteHorizonControlSystem):
  def __init__(self):
    self.b = 0.525
    self.d = 0.5
    self.c = 0.0001
    self.e = 0.5

    self.g = 0.1
    self.a = 0.2

    self.S_0 = 1000.0
    self.E_0 = 100.0
    self.I_0 = 50.0
    self.R_0 = 15.0
    self.N_0 = self.S_0 + self.E_0 + self.I_0 + self.R_0

    self.A = 0.1
    self.M = 1000

    super().__init__(
      _type = SystemType.SEIR,
      x_0 = jnp.array([self.S_0, self.E_0, self.I_0, self.N_0]),
      x_T = None,
      T = 20,
      bounds = jnp.array([
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [-jnp.inf, jnp.inf],
        [0.0, 1.0],
      ]),
      terminal_cost = False,
    )

  def dynamics(self, y_t: jnp.ndarray, u_t: jnp.float64) -> jnp.ndarray:
    S, E, I, N = y_t

    Ṡ = jnp.squeeze(self.b*N - self.d*S - self.c*S*I - u_t*S)
    Ė = jnp.squeeze(self.c*S*I - (self.e+self.d)*E)
    İ = jnp.squeeze(self.e*E - (self.g+self.a+self.d)*I)
    Ṅ = jnp.squeeze((self.b-self.d)*N - self.a*I)

    ẏ_t = jnp.array([Ṡ, Ė, İ, Ṅ])
    return ẏ_t
  
  def cost(self, y_t: jnp.ndarray, u_t: jnp.float64, t: float = None) -> jnp.float64:
    return self.A * y_t[2] + u_t ** 2

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    sns.set()
    plt.figure(figsize=(12, 2.5))
    ts_x = jnp.linspace(0, self.T, x.shape[0])
    ts_u = jnp.linspace(0, self.T, u.shape[0])

    plt.subplot(151)
    plt.title('applied control')
    plt.plot(ts_u, u)
    plt.ylim(-0.1, 1.01)

    for idx, title in enumerate(['S', 'E', 'I', 'N']):
      plt.subplot(1, 5, idx+2)
      plt.title(title)
      plt.plot(ts_x, x[:, idx])

    plt.tight_layout()
    plt.show()


class Tumour(FiniteHorizonControlSystem):
  # Practical Methods for Optimal Control Using Nonlinear Programming (Third Edition, Chapter 8.17)
  def __init__(self):
    # Parameters
    self.ξ = 0.084 # per day (tumour growth)
    self.b = 5.85 # per day (birth rate)
    self.d = 0.00873 # per mm^2 per day (death rate)
    self.G = 0.15 # kg per mg of dose per day (antiangiogenic killing)
    self.mu = 0.02 # per day (loss of endothelial cells due to natural causes)
    t_F = 1.2 # days
    # State and Control Bounds
    a = 75 # maximum instantaneous dosage
    A = 15 # maximum cumulative dosage
    p_ = q_ = ((self.b-self.mu)/self.d)**(3/2) # asymptotically stable focus
    # Initial State
    p_0 = p_ / 2 # Initial tumour volume
    q_0 = q_ / 4 # Initial vascular capacity
    y_0 = 0 # Initial cumulative dosage
    assert p_0 >= q_0 # condition for well-posed problem
    super().__init__(
      _type = SystemType.TUMOUR,
      x_0 = jnp.array([p_0, q_0, y_0]),
      x_T = None,
      T = t_F,
      bounds = jnp.array([
        [0, p_], # p
        [0, q_], # q
        [0, A], # y
        [0, a], # control
      ]),
      terminal_cost = True,
    )

  def dynamics(self, x_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
    p, q, y = x_t
    _p = jnp.squeeze(-self.ξ * p * jnp.log(p/q))
    _q = jnp.squeeze(q * (self.b - (self.mu + self.d * p**(2/3) + self.G * u_t)))
    _y = jnp.squeeze(u_t)
    return jnp.asarray([_p, _q, _y])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return 0

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: jnp.ndarray, T: jnp.ndarray = None) -> float:
    p, q, y = x_T
    return p

  def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
    colnames = ['p', 'q', 'y']
    x = pd.DataFrame(x, columns=colnames)

    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 3))
    ts_x = jnp.linspace(0, self.T, x.shape[0])
    ts_u = jnp.linspace(0, self.T, u.shape[0])

    for idx, title in enumerate(colnames):
      plt.subplot(1, 4, idx+1)
      plt.title(title)
      plt.plot(ts_x, x[title])
      plt.xlabel('time (days)')

    plt.subplot(1, 4, 4)
    plt.title('u')
    plt.step(ts_u, u, where="post")
    plt.xlabel('time (days)')

    plt.tight_layout()
    plt.show()
