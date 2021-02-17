from typing import Union, Optional
import gin

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from source.systems import IndirectFHCS


@gin.configurable
class Glucose(IndirectFHCS):
  def __init__(self, a=1., b=1., c=1., A=2., l=.5, x_0=(.75, 0.), T=.2):
    """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 16, Lab 10)
    Model is presented in more details in Martin Eisen. Mathematical Methods and Models in the Biological Sciences.
    Prentice Hall, Englewood Cliffs, New Jersey, 1988.

    This environment models the blood glucose (x_0(t0) level of a diabetic person in the presence of injected
    insulin (u(t)), along with the net hormonal concentration (x_1(t)) of insulin in the person's system.
    In this model, the diabetic person is assumed to be unable to produce natural insulin via their pancreas.

    Note that the model was developed for regulating blood glucose levels over a short window of time. As such, T
    should be kept under 0.45 in order for the model to make sense.
    (T is measured in days, so 0.45 corresponds to ~11 hours)

    The goal of the control is to maintain the blood glucose level close to a desired level, l, while also taking
    into account that there is a cost associated to the treatment. Thus the objective is:

    .. math::

      \min_{u} \quad &\int_0^T A(x_0(t)-l)^2 + u_f(t)^2  dt \\
      \mathrm{s.t.}\qquad & x_0'(t) = -ax_0(t) - bx_1(t) ,\; x_0(0) > 0 \\
      & x_1'(t) = -cx_1(t) + u(t) ,\; x_1(0)=0 \\
      & a,b,c > 0 \; A \geq 0

    :param a: Rate of decrease in glucose level resulting of its use by the body
    :param b: Rate of decrease in glucose level resulting from its degradation provoked by insulin
    :param c: Rate of degradation of the insulin
    :param A: Weight parameter balancing the objective
    :param l: Desired level of blood glucose
    :param x_0: Initial blood glucose level and insulin level (x_0,x_1)
    :param T: Horizon (Should be kept under 0.45)
    """
    super().__init__(
      x_0=jnp.array([
        x_0[0],
        x_0[1],
      ]),                     # Starting state
      x_T=None,               # Terminal state, if any
      T=T,                    # Duration of experiment
      bounds=jnp.array([      # Bounds over the states (x_0, x_1 ...) are given first,
        [jnp.NINF, jnp.inf],  # followed by bounds over controls (u_0,u_1,...)
        [jnp.NINF, jnp.inf],
        [0, jnp.inf],
      ]),
      terminal_cost=False,
      discrete=False,
    )

    self.adj_T = None  # Final condition over the adjoint, if any
    self.a = a
    self.b = b
    self.c = c
    self.A = A
    self.l = l

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x_0, x_1 = x_t
    if u_t.ndim > 0:
      u_t, = u_t

    d_x = jnp.array([
      -self.a*x_0 - self.b*x_1,
      -self.c*x_1 + u_t
      ])

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return self.A*(x_t[0]-self.l)**2 + u_t**2

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return jnp.array([
      -2*self.A*(x_t[0]-self.l) + adj_t[0]*self.a,
      adj_t[0]*self.b + adj_t[1]*self.c
    ])

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char_0 = -adj_t[:, 1]/2
    char_0 = char_0.reshape(-1, 1)

    return char_0