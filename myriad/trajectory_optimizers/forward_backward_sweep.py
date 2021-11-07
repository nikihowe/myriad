# (c) 2021 Nikolaus Howe
import jax.numpy as jnp

from jax.flatten_util import ravel_pytree
from jax.ops import index_update
# from ipopt import minimize_ipopt
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

from myriad.config import Config, HParams, OptimizerType, SystemType, IntegrationMethod, QuadratureRule
from myriad.custom_types import Solution
from myriad.nlp_solvers import solve
from myriad.systems import FiniteHorizonControlSystem, IndirectFHCS
from myriad.utils import integrate_in_parallel, integrate_time_independent, \
  integrate_time_independent_in_parallel, integrate_fbsm
from myriad.trajectory_optimizers.base import IndirectMethodOptimizer


class FBSM(IndirectMethodOptimizer):  # Forward-Backward Sweep Method
  """
  The Forward-Backward Sweep Method, as described in Optimal Control Applied to Biological Models, Lenhart & Workman
  An iterative solver that, given an initial guess over the controls, will do a forward pass to retrieve the state
  variables trajectory followed by a backward pass to retrieve the adjoint variables trajectory. The optimality
  characterization is then used to update the control values.
  The process is repeated until convergence over the controls.
  """

  def __init__(self, hp: HParams, cfg: Config, system: IndirectFHCS):
    self.system = system
    self.N = hp.fbsm_intervals
    self.h = system.T / self.N
    if system.discrete:
      self.N = int(system.T)
      self.h = 1
    state_shape = system.x_0.shape[0]
    control_shape = system.bounds.shape[0] - state_shape

    x_guess = jnp.vstack((system.x_0, jnp.zeros((self.N, state_shape))))
    if system.discrete:
      u_guess = jnp.zeros((self.N, control_shape))
    else:
      u_guess = jnp.zeros((self.N + 1, control_shape))
    if system.adj_T is not None:
      adj_guess = jnp.vstack((jnp.zeros((self.N, state_shape)), system.adj_T))
    else:
      adj_guess = jnp.zeros((self.N + 1, state_shape))
    self.t_interval = jnp.linspace(0, system.T, num=self.N + 1).reshape(-1, 1)

    guess, unravel = ravel_pytree((x_guess, u_guess, adj_guess))
    self.x_guess, self.u_guess, self.adj_guess = x_guess, u_guess, adj_guess

    x_bounds = system.bounds[:-1]
    u_bounds = system.bounds[-1:]
    bounds = jnp.vstack((x_bounds, u_bounds))
    self.x_bounds, self.u_bounds = x_bounds, u_bounds

    # Additional condition if terminal condition are present
    self.terminal_cdtion = False
    if self.system.x_T is not None:
      num_term_state = 0
      for idx, x_Ti in enumerate(self.system.x_T):
        if x_Ti is not None:
          self.terminal_cdtion = True
          self.term_cdtion_state = idx
          self.term_value = x_Ti
          num_term_state += 1
        if num_term_state > 1:
          raise NotImplementedError("Multiple states with terminal condition not supported yet")

    super().__init__(hp, cfg, bounds, guess, unravel)

  def reinitiate(self, a):
    """Helper function for `sequencesolver`
    """
    state_shape = self.system.x_0.shape[0]
    control_shape = self.system.bounds.shape[0] - state_shape

    self.x_guess = jnp.vstack((self.system.x_0, jnp.zeros((self.N, state_shape))))
    self.u_guess = jnp.zeros((self.N + 1, control_shape))
    if self.system.adj_T is not None:
      adj_guess = jnp.vstack((jnp.zeros((self.N, state_shape)), self.system.adj_T))
    else:
      adj_guess = jnp.zeros((self.N + 1, state_shape))
    self.adj_guess = index_update(adj_guess, (-1, self.term_cdtion_state), a)

  def solve(self) -> Solution:
    """Solve the continuous optimal problem with the Forward-Backward Sweep Method"""
    if self.terminal_cdtion:
      return self.sequencesolver()
    n = 0
    while n == 0 or self.stopping_criterion((self.x_guess, old_x), (self.u_guess, old_u), (self.adj_guess, old_adj)):
      old_u = self.u_guess.copy()
      old_x = self.x_guess.copy()
      old_adj = self.adj_guess.copy()

      self.x_guess = integrate_fbsm(self.system.dynamics, self.x_guess[0], self.u_guess, self.h, self.N,
                                    t=self.t_interval, discrete=self.system.discrete)[-1]
      self.adj_guess = integrate_fbsm(self.system.adj_ODE, self.adj_guess[-1], self.x_guess, -1 * self.h, self.N,
                                      self.u_guess, t=self.t_interval, discrete=self.system.discrete)[-1]

      u_estimate = self.system.optim_characterization(self.adj_guess, self.x_guess, self.t_interval)
      # Use basic convex approximation to update the guess on u
      self.u_guess = 0.5 * (u_estimate + old_u)

      n = n + 1

    solution = {
      'x': self.x_guess,
      'u': self.u_guess,
      'adj': self.adj_guess
    }
    return solution

  def sequencesolver(self) -> Solution:
    """Implement the secant method for the special case where there is a terminal value on some state variables in
    addition to the initial values.
    """
    self.terminal_cdtion = False
    count = 0

    # Adjust lambda to the initial guess
    a = self.system.guess_a
    self.reinitiate(a)
    tmp_solution = self.solve()
    x_a = tmp_solution['x']
    # x_a, _, _ = self.solve()
    Va = x_a[-1, self.term_cdtion_state] - self.term_value
    b = self.system.guess_b
    self.reinitiate(b)
    tmp_solution = self.solve()
    x_b = tmp_solution['x']
    # x_b, _, _ = self.solve()
    Vb = x_b[-1, self.term_cdtion_state] - self.term_value

    while jnp.abs(Va) > 1e-10:
      if jnp.abs(Va) > jnp.abs(Vb):
        a, b = b, a
        Va, Vb = Vb, Va

      d = Va * (b - a) / (Vb - Va)
      b = a
      Vb = Va
      a = a - d
      self.reinitiate(a)
      tmp_solution = self.solve()
      x_a = tmp_solution['x']
      # x_a, _, _ = self.solve()
      Va = x_a[-1, self.term_cdtion_state] - self.term_value
      count += 1

    solution = {
      'x': self.x_guess,
      'u': self.u_guess,
      'adj': self.adj_guess
    }
    return solution
