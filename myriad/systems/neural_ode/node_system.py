# (c) 2021 Nikolaus Howe
from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
  from myriad.neural_ode.create_node import NeuralODE

import jax.numpy as jnp

from myriad.systems.base import FiniteHorizonControlSystem
from myriad.custom_types import Control, Cost, DState, Params, State, Timestep


class NodeSystem(FiniteHorizonControlSystem):
  def __init__(self, node: NeuralODE, true_system: FiniteHorizonControlSystem) -> None:
    """
    A generic system with NODE dynamics
    """
    self.node = node
    self.true_system = true_system

    super().__init__(
      x_0=true_system.x_0,
      x_T=true_system.x_T,
      T=true_system.T,
      bounds=true_system.bounds,
      terminal_cost=true_system.terminal_cost
    )
  # NOTE: for now, only the dynamics is learnable (not the cost)

  # True dynamics
  def dynamics(self, x_t: State, u_t: Control, t: Timestep = None) -> DState:
    return self.true_system.dynamics(x_t, u_t)  # TODO: we really should make the dynamics accept a t

  # Neural ODE dynamics
  def parametrized_dynamics(self, params: Params, x_t: State, u_t: Control, t: Timestep = None) -> DState:
    x_and_u = jnp.append(x_t, u_t)
    return self.node.net.apply(params, x_and_u)

  # True cost
  def cost(self, x_t: State, u_t: Control, t: Timestep = None) -> Cost:
    return self.true_system.cost(x_t, u_t, t)
