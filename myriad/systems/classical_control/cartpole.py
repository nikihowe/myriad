# (c) 2021 Nikolaus Howe
import jax.numpy as jnp

from typing import Optional

from myriad.systems.base import FiniteHorizonControlSystem
from myriad.custom_types import Control, Cost, DState, Params, State, Timestep


class CartPole(FiniteHorizonControlSystem):
  """
  Cart-pole swing-up, from [(Kelly, 2017)](https://epubs.siam.org/doi/10.1137/16M1062569).

  This environment models a cart moving in a unidimensional direction with a pendulum hanging freely from it.
  The usually associated task is to move the cart in such a way that the pendulum swings up to the non-stationary
  equilibrium point above the cart.

  The goal is to find the cart velocity ( \\(q_1'(t)\\) ) that will impede
  an angular velocity of the pendulum ( \\(q_2'(t)\\) ) that achieves the task, while minimising the force
  ( \\(u(t)\\) ) needed to generate such a movement. The system to solve is:

  .. math::

      \\begin{align}
      & \\min_{u} \\quad && \\int_0^T u(t)^2  dt \\\\
      & \\; \\mathrm{s.t.}\\quad && q_1''(t) = \\frac{l m_2 \\sin(q_2(t)) q_2^2(t)' + u(t) + m_2 g \\cos(q_2(t)) \\sin(q_2(t))}
      {m_1 + m_2 (1-\\cos^2 (q_2(t)))}\\\\
      & && q_2''(t) = - \\frac{l m_2 \\cos(q_2(t))\\sin(q_2(t)) q_2^2(t) + u(t) \\cos(q_2(t))
      + (m_1 +m_2)g \\sin(q_2(t))}{l m_1 + l m_2 (1 - \\cos^2(q_2(t))} \\\\
      & && q_1(0)=q_2(0)=q_1'(0)=q_2'(0)=0 \\\\
      & && q_1(T) = d ,\\; q_2(T) = \\pi ,\\; q_1'(T)=q_2'(T)=0 \\\\
      & && -d_M <= q_1(t) <= d_M ,\\; -u_M <= u(t) <= u_M
      \\end{align}

  Notes
    -----
    \\(q_1\\): Position of the cart \n
    \\(q_2\\): Angle of the pole \n
    \\(q_1'\\): Velocity of the cart \n
    \\(q_2'\\): Angular velocity of the pole \n
    \\(m_1\\): Mass of the cart \n
    \\(m_2\\): Mass of the pendulum \n
    \\(l\\): Length of the pole \n
    \\(g\\): Gravity force \n
    \\(d_M\\): Maximal distance that can be traveled by the cart \n
    \\(u_M\\): Maximal force that can be applied to the motor \n
    \\(T\\): The horizon
  """

  def __init__(self, g: float = 9.81, m1: float = 1., m2: float = .3, length: float = 0.5):
    # Physical parameters for the cart-pole example (Table 3)
    self.m1 = m1  # kg mass of cart
    self.m2 = m2  # kg mass of pole
    self.length = length  # m pole length
    self.g = g  # m/s^2 gravity acceleration

    self.u_max = 20  # N maximum actuator force
    self.d_max = 2.0  # m extent of the rail that cart travels on
    self.d = 1.0  # m distance traveled during swing-up

    super().__init__(
      x_0=jnp.array([0., 0., 0., 0.]),  # Starting state (Eq. 6.9)
      x_T=jnp.array([self.d, jnp.pi, 0., 0.]),  # Ending state (Eq. 6.9)
      T=2.0,  # s duration of swing-up,
      bounds=jnp.array([
        [-self.d_max, self.d_max],  # Eq. 6.7
        [-2 * jnp.pi, 2 * jnp.pi],
        [-5., 5.],  # Observed from optimal plot, taken as reasonable
        [-10., 10.],
        [-self.u_max, self.u_max],  # Control bounds (Eq. 6.8)
      ]),
      terminal_cost=False,
    )

  # Cart-Pole Example: System Dynamics (Section 6.1)
  def dynamics(self, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> DState:
    x, theta, dx, dtheta = x_t
    # Eq. 6.1
    ddx = ((self.length * self.m2 * jnp.sin(theta) * dtheta ** 2 + u_t + self.m2 * self.g * jnp.cos(theta) * jnp.sin(theta))
           / (self.m1 + self.m2 * (1 - jnp.cos(theta) ** 2)))
    ddx = jnp.squeeze(ddx)
    # Eq. 6.2
    ddtheta = - ((self.length * self.m2 * jnp.cos(theta) * dtheta ** 2 + u_t * jnp.cos(theta)
                  + (self.m1 + self.m2) * self.g * jnp.sin(theta))
                 / (self.length * self.m1 + self.length * self.m2 * (1 - jnp.cos(theta) ** 2)))
    ddtheta = jnp.squeeze(ddtheta)
    return jnp.array([dx, dtheta, ddx, ddtheta])

  def parametrized_dynamics(self, params: Params, x_t: State, u_t: Control, t: Optional[Timestep] = None) -> DState:
    g = jnp.abs(params['g'])  # convert negative values to positive ones
    m1 = jnp.abs(params['m1'])
    m2 = jnp.abs(params['m2'])
    length = jnp.abs(params['length'])
    x, theta, dx, dtheta = x_t
    # Eq. 6.1
    ddx = ((length * m2 * jnp.sin(theta) * dtheta ** 2 + u_t + m2 * g * jnp.cos(theta) * jnp.sin(theta))
           / (m1 + m2 * (1 - jnp.cos(theta) ** 2)))
    ddx = jnp.squeeze(ddx)
    # Eq. 6.2
    ddtheta = - ((length * m2 * jnp.cos(theta) * dtheta ** 2 + u_t * jnp.cos(theta)
                  + (m1 + m2) * g * jnp.sin(theta))
                 / (length * m1 + length * m2 * (1 - jnp.cos(theta) ** 2)))
    ddtheta = jnp.squeeze(ddtheta)
    return jnp.array([dx, dtheta, ddx, ddtheta])

  def cost(self, x_t: State, u_t: Control, t: Timestep = None) -> Cost:
    # Eq. 6.3
    return u_t ** 2

  def parametrized_cost(self, params: Params, x_t: State, u_t: Control, t: Optional[Timestep]) -> Cost:
    return self.cost(x_t, u_t, t)

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   x = pd.DataFrame(x, columns=['q1', 'q2', 'q̈1', 'q̈2'])
  #
  #   # Plot optimal trajectory (Figure 10)
  #   sns.set(style='darkgrid')
  #   plt.figure(figsize=(9, 6))
  #   ts_x = jnp.linspace(0, self.T, x.shape[0])
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   plt.subplot(3, 1, 1)
  #   plt.ylabel('position (m)')
  #   plt.xlim(0, 2.01)
  #   plt.ylim(0, 1.5)
  #   plt.plot(ts_x, x['q1'], '-bo', clip_on=False, zorder=10)
  #
  #   plt.subplot(3, 1, 2)
  #   plt.ylabel('angle (rad)')
  #   plt.plot(ts_x, x['q2'], '-bo', clip_on=False, zorder=10)
  #   plt.xlim(0, 2.01)
  #   plt.ylim(-2, 4)
  #
  #   plt.subplot(3, 1, 3)
  #   plt.ylabel('force (N)')
  #   # plt.plot(ts_u, u, '-bo', clip_on=False, zorder=10)
  #   plt.step(ts_u, u, where="post", clip_on=False)
  #   plt.xlim(0, 2.01)
  #   plt.ylim(-20, 11)
  #
  #   plt.xlabel('time (s)')
  #   plt.tight_layout()
  #   plt.show()
