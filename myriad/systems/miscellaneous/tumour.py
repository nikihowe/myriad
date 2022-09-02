import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from myriad.custom_types import Params
from myriad.systems import FiniteHorizonControlSystem


class Tumour(FiniteHorizonControlSystem):
  """
  Tumour anti-angiogenesis model, from [Practical Methods for Optimal Control Using Nonlinear Programming (Third Edition, Chapter 10.70)](https://my.siam.org/Store/Product/viewproduct/?ProductId=31657301).
  More details can be found in [Ledzewicz and Schattler](https://www.siue.edu/~uledzew/papers/angioMTNS.pdf)

  The model describes the growth of a tumor that needs its own blood vessels to continue to grow. Endothelial cells
  provide lining for newly forming blood vessels and as such, can be inhibited to reduce the tumor growth. The model can
  be described as:

  .. math::

    \\begin{align}
    & \\min_{u} \\quad && p(T) \\\\
    & \\; \\mathrm{s.t.}\\quad && p'(t) = -\\xi p \\ln(\\frac{p}{q}) \\\\
    & && q'(t) = bp - (\\mu + dp^{\\frac{2}{3}}) q - G u q \\\\
    & && y'(t) = u \\\\
    & && p(0) = p_0 ,\\; q(0) = q_0 ,\\; y(0) = 0  \\\\
    & && 0 <= p(t) ,\\; 0 <= q(t) ,\\; 0 <= y(t) <= A ,\\; 0 <= u(t) <= a \\\\
    \\end{align}

  Notes
  -----
  \\(p(t)\\): The size of the tumor \n
  \\(q(t)\\): The amount of vascular endothelial cells \n
  \\(u(t)\\): Angionesic dose rate; an external inhibitor decreasing \\(q(t)\\) \n
  \\(y(t)\\) : A measure of the total amount of external inhibitor used \n
  \\(a\\): Instantaneous limit over the inhibitor that can be administered \n
  \\(A\\): Total limit over the inhibitor that can be administered \n
  \\(\\xi\\): Tumor growth parameter \n
  \\(\\mu\\): Loss rate of endothelial cells from natural causes \n
  \\(b\\): Birth rate of endothelial cells from stimulation by the tumor \n
  \\(d\\): Death rate of endothelial cells from inhibition by the tumor
  """

  def __init__(self, xi=0.084, b=5.85, d=0.00873, G=0.15, mu=0.02):
    # Learnable parameters
    self.xi = xi  # per day (tumour growth)
    self.b = b  # per day (birth rate)
    self.d = d  # per mm^2 per day (death rate)
    self.G = G  # kg per mg of dose per day (antiangiogenic killing)
    self.mu = mu  # per day (loss of endothelial cells due to natural causes)

    t_F = 1.2  # days

    # State and Control Bounds
    a = 75  # maximum instantaneous dosage
    A = 15  # maximum cumulative dosage
    p_ = q_ = ((self.b - self.mu) / self.d) ** (3 / 2)  # asymptotically stable focus

    # Initial State
    p_0 = p_ / 2  # Initial tumour volume
    q_0 = q_ / 4  # Initial vascular capacity
    y_0 = 0  # Initial cumulative dosage
    assert p_0 >= q_0  # condition for well-posed problem
    super().__init__(
      x_0=jnp.array([p_0, q_0, y_0]),
      x_T=None,
      T=t_F,
      bounds=jnp.array([
        [0., p_],  # p
        [0., q_],  # q
        [0., A],  # y
        [0., a],  # control
      ]),
      terminal_cost=True,
    )

  def dynamics(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    p, q, y = x_t
    _p = jnp.squeeze(-self.xi * p * jnp.log(p / q))
    _q = jnp.squeeze(q * (self.b - (self.mu + self.d * p ** (2 / 3) + self.G * u_t)))
    _y = jnp.squeeze(u_t)
    return jnp.asarray([_p, _q, _y])

  def parametrized_dynamics(self, params: Params, x_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    xi = params['xi']
    b = params['b']
    d = params['d']
    mu = params['mu']
    G = params['G']

    p, q, y = x_t
    _p = jnp.squeeze(-xi * p * jnp.log(p / q))
    _q = jnp.squeeze(q * (b - (mu + d * p ** (2 / 3) + G * u_t)))
    _y = jnp.squeeze(u_t)
    return jnp.asarray([_p, _q, _y])

  def cost(self, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    # nh: I think this should be changed to u^2, otherwise there
    # is no penalty for oscillating in u
    # return u_t * u_t
    return 0.

  def parametrized_cost(self, params: Params, x_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return 0.  # nothing to learn here

  def terminal_cost_fn(self, x_T: jnp.ndarray, u_T: jnp.ndarray, T: jnp.ndarray = None) -> float:
    p, q, y = x_T
    return p

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   colnames = ['p', 'q', 'y']
  #   x = pd.DataFrame(x, columns=colnames)
  #
  #   sns.set(style='darkgrid')
  #   plt.figure(figsize=(10, 3))
  #   ts_x = jnp.linspace(0, self.T, x.shape[0])
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   for idx, title in enumerate(colnames):
  #     plt.subplot(1, 4, idx+1)
  #     plt.title(title)
  #     plt.plot(ts_x, x[title])
  #     plt.xlabel('time (days)')
  #
  #   plt.subplot(1, 4, 4)
  #   plt.title('u')
  #   plt.step(ts_u, u, where="post")
  #   plt.xlabel('time (days)')
  #
  #   plt.tight_layout()
  #   plt.show()
