from dataclasses import dataclass
from typing import Optional

import jax.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import SystemType, HParams


@dataclass
class FiniteHorizonControlSystem(object):
  x_0: np.array # state at time 0
  x_T: Optional[np.array] # state at time T
  T: float # duration of trajectory
  bounds: np.ndarray # State and control bounds
  terminal_cost: bool # Whether only the final state and control are inputs to the cost

  # def __post_init__(self):
  #   self.x_0 = self.x_0.astype(np.float64)
  #   if self.x_T is not None:
  #     assert self.x_0.shape == self.x_T.shape
  #     self.x_T = self.x_T.astype(np.float64)
  #   assert self.bounds.shape == (self.x_0.shape[0]+1, 2)
  #   assert self.T > 0

  def dynamics(self, x_t: np.ndarray, u_t: float, v_t: np.ndarray) -> np.ndarray:
    raise NotImplementedError
  
  def cost(self, x_t: np.ndarray, u_t: float) -> float:
    raise NotImplementedError

  def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: Optional[np.array]) -> None:
    raise NotImplementedError


def get_system(hp: HParams) -> FiniteHorizonControlSystem:
  from .LenhartSystems.Lab1 import Lab1 #TODO: This is just dirty : to clean
  from .LenhartSystems.Lab5 import Lab5
  from .LenhartSystems.Lab6 import Lab6
  from .LenhartSystems.Lab7 import Lab7
  from .LenhartSystems.Lab8 import Lab8
  from .LenhartSystems.Lab9 import Lab9
  if hp.system == SystemType.CARTPOLE:
    return CartPole()
  elif hp.system == SystemType.VANDERPOL:
    return VanDerPol()
  elif hp.system == SystemType.SEIR:
    return SEIR()
  elif hp.system == SystemType.TUMOUR:
    return Tumour()
  elif hp.system == SystemType.LENHART1:
    return Lab1()
  elif hp.system == SystemType.LENHART5:
    return Lab5()
  elif hp.system == SystemType.LENHART6:
    return Lab6()
  elif hp.system == SystemType.LENHART7:
    return Lab7()
  elif hp.system == SystemType.LENHART8:
    return Lab8()
  elif hp.system == SystemType.LENHART9:
    return Lab9()
  else:
    raise KeyError


class CartPole(FiniteHorizonControlSystem):
  def __init__(self):
    # Physical parameters for the cart-pole example (Table 3)
    self.m1 = 1.0 #kg mass of cart
    self.m2 = 0.3 #kg mass of pole
    self.l = 0.5 #m pole length
    self.g = 9.81 #m/s^2 gravity acceleration
    self.u_max = 20 #N maximum actuator force
    self.d_max = 2.0 #m extent of the rail that cart travels on
    self.d = 1.0 #m distance traveled during swing-up

    super().__init__(
      x_0 = np.zeros(4), # Starting state (Eq. 6.9)
      x_T = np.array([self.d,np.pi,0,0]), # Ending state (Eq. 6.9)
      T = 2.0, #s duration of swing-up,
      bounds = np.array([
        [-self.d_max, self.d_max], # Eq. 6.7
        [-2*np.pi, 2*np.pi],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [-self.u_max, self.u_max], # Control bounds (Eq. 6.8)
      ]),
      terminal_cost = False,
    )

  # Cart-Pole Example: System Dynamics (Section 6.1)
  def dynamics(self, x_t: np.ndarray, u_t: float) -> np.ndarray:
    q1, q2, q̇1, q̇2 = x_t
    # Eq. 6.1
    q̈1 = (self.l * self.m2 * np.sin(q2) * q̇2**2 + u_t + self.m2 * self.g * np.cos(q2) * np.sin(q2)) / (self.m1 + self.m2 * (1 - np.cos(q2)**2))
    q̈1 = np.squeeze(q̈1)
    # Eq. 6.2
    q̈2 = - (self.l * self.m2 * np.cos(q2) * q̇2**2 + u_t * np.cos(q2) + (self.m1 + self.m2) * self.g * np.sin(q2)) / (self.l * self.m1 + self.l * self.m2 * (1 - np.cos(q2)**2))
    q̈2 = np.squeeze(q̈2)
    return np.array([q̇1, q̇2, q̈1, q̈2])
  
  def cost(self, x_t: np.ndarray, u_t: float) -> float:
    # Eq. 6.3
    return u_t ** 2
  
  def plot_solution(self, x: np.ndarray, u: np.ndarray) -> None:
    x = pd.DataFrame(x, columns=['q1','q2','q̈1','q̈2'])

    # Plot optimal trajectory (Figure 10)
    sns.set(style='darkgrid')
    plt.figure(figsize=(9,6))
    ts_x = np.linspace(0, self.T, x.shape[0])
    ts_u = np.linspace(0, self.T, u.shape[0])

    plt.subplot(3,1,1)
    plt.ylabel('position (m)')
    plt.xlim(0,2)
    plt.ylim(0,1.5)
    plt.plot(ts_x, x['q1'], '-bo', clip_on=False, zorder=10)

    plt.subplot(3,1,2)
    plt.ylabel('angle (rad)')
    plt.plot(ts_x, x['q2'], '-bo', clip_on=False, zorder=10)
    plt.xlim(0,2)
    plt.ylim(-2,4)

    plt.subplot(3,1,3)
    plt.ylabel('force (N)')
    # plt.plot(ts_u, u, '-bo', clip_on=False, zorder=10)
    plt.step(ts_u, u, where="post", clip_on=False)
    plt.xlim(0,2)
    plt.ylim(-20,10)

    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()

    
class VanDerPol(FiniteHorizonControlSystem):
  def __init__(self):
    super().__init__(
      x_0 = np.array([0., 1.]),
      x_T = np.zeros(2),
      T = 10.0,
      bounds = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [-0.75, 1.0],
      ]),
      terminal_cost = False,
    )

  def dynamics(self, x_t: np.ndarray, u_t: float) -> np.ndarray:
    x0, x1 = x_t
    _x0 = np.squeeze((1. - x1**2) * x0 - x1 + u_t)
    _x1 = np.squeeze(x0)
    return np.asarray([_x0, _x1])
  
  def cost(self, x_t: np.ndarray, u_t: float) -> float:
    return x_t.T @ x_t + u_t ** 2

  def plot_solution(self, x: np.ndarray, u: np.ndarray) -> None:
    x = pd.DataFrame(x, columns=['x0','x1'])

    sns.set(style='darkgrid')
    plt.figure(figsize=(9,4))
    ts_u = np.linspace(0, self.T, u.shape[0])

    plt.subplot(1,2,1)
    plt.plot(x['x0'], x['x1'])
    
    plt.subplot(1,2,2)
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

    self.S_0 = 1000
    self.E_0 = 100
    self.I_0 = 50
    self.R_0 = 15
    self.N_0 = self.S_0 + self.E_0 + self.I_0 + self.R_0

    self.A = 0.1
    self.M = 1000

    super().__init__(
      x_0 = np.array([self.S_0, self.E_0, self.I_0, self.N_0]),
      x_T = None,
      T = 20,
      bounds = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [0.0, 1.0],
      ]),
      terminal_cost = False,
    )

  def dynamics(self, y_t: np.ndarray, u_t: np.float64) -> np.ndarray:
    S, E, I, N = y_t

    Ṡ = np.squeeze(self.b*N - self.d*S - self.c*S*I - u_t*S)
    Ė = np.squeeze(self.c*S*I - (self.e+self.d)*E)
    İ = np.squeeze(self.e*E - (self.g+self.a+self.d)*I)
    Ṅ = np.squeeze((self.b-self.d)*N - self.a*I)

    ẏ_t = np.array([Ṡ, Ė, İ, Ṅ])
    return ẏ_t
  
  def cost(self, y_t: np.ndarray, u_t: np.float64) -> np.float64:
    return self.A * y_t[2] + u_t ** 2

  def plot_solution(self, x: np.ndarray, u: np.ndarray) -> None:
    sns.set()
    plt.figure(figsize=(12,2.5))
    ts_x = np.linspace(0, self.T, x.shape[0])
    ts_u = np.linspace(0, self.T, u.shape[0])

    plt.subplot(151)
    plt.title('applied control')
    plt.plot(ts_u, u)
    plt.ylim(-0.1, 1.01)

    for idx, title in enumerate(['S', 'E', 'I', 'N']):
      plt.subplot(1,5,idx+2)
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
      x_0 = np.array([p_0, q_0, y_0]),
      x_T = None,
      T = t_F,
      bounds = np.array([
        [0, p_], # p
        [0, q_], # q
        [0, A], # y
        [0, a], # control
      ]),
      terminal_cost = True,
    )

  def dynamics(self, x_t: np.ndarray, u_t: float) -> np.ndarray:
    p, q, y = x_t
    _p = np.squeeze(-self.ξ * p * np.log(p/q))
    _q = np.squeeze(q * (self.b - (self.mu + self.d * p**(2/3) + self.G * u_t)))
    _y = np.squeeze(u_t)
    return np.asarray([_p, _q, _y])
  
  def cost(self, x_t: np.ndarray, u_t: float) -> float:
    p, q, y = x_t
    return p

  def plot_solution(self, x: np.ndarray, u: np.ndarray) -> None:
    colnames = ['p','q','y']
    x = pd.DataFrame(x, columns=colnames)

    sns.set(style='darkgrid')
    plt.figure(figsize=(10,3))
    ts_x = np.linspace(0, self.T, x.shape[0])
    ts_u = np.linspace(0, self.T, u.shape[0])

    for idx, title in enumerate(colnames):
      plt.subplot(1,4,idx+1)
      plt.title(title)
      plt.plot(ts_x, x[title])
      plt.xlabel('time (days)')

    plt.subplot(1,4,4)
    plt.title('u')
    plt.step(ts_u, u, where="post")
    plt.xlabel('time (days)')

    plt.tight_layout()
    plt.show()
