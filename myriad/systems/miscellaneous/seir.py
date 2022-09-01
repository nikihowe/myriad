import jax.numpy as jnp

from myriad.systems import FiniteHorizonControlSystem


class SEIR(FiniteHorizonControlSystem):
  """
  SEIR epidemic model for COVID-19, inspired by [Perkins and Espana, 2020](https://link.springer.com/article/10.1007/s11538-020-00795-y).

  This model is an adaptation of SEIR models, specifically tailored to COVID-19 epidemic trying to limit the spread
  via non-pharmaceutical interventions (example: reducing contacts between individuals). As such, the control variable
  ( \\(u(t)\\) ) is a reduction in the transmission coefficient ( \\( \\beta\\) ) resulting from all societal measures
  that allow to control the virus spread. The goal of the model is to help decision-maker quantify the impact of
  policies limiting the spread.

  The formal model is given by:

  .. math::

    \\begin{align}
    & \\min_{u} \\quad && \\int_0^T D(t)^2 + cu(t)^2 dt \\\\
    & \\; \\mathrm{s.t.}\\quad && S'(t) = \\mu - (\\delta + \\beta(1-u(t))(\\alpha A(t) + I(t) + H(t))
     + \\iota + \\nu)S(t) \\\\
    & && E'(t) = ( \\beta(1-u(t))(\\alpha A(t) + I(t) + H(t))) (S(t) + (1-\\epsilon)V(t)) + \\iota S(t)
     - (\\delta + \\rho) E(t) \\\\
    & && A'(t) = (1-\\sigma)\\rho E(t) - (\\delta + \\gamma) A(t) \\\\
    & && I'(t) = \\sigma \\rho E(t) - (\\delta + \\gamma)I(t) \\\\
    & && H'(t) = \\gamma \\kappa I(t) - (\\delta + \\eta) H(t) \\\\
    & && V'(t) = \\nu S(t) - (\\delta + \\beta(1 -u(t))(\\alpha A(t) + I(t) + H(t)) (1-\\epsilon)) V(t) \\\\
    & && S(0) = S_0 ,\\; E(0) = E_0 ,\\; A(0) = A_0 ,\\; I(0) = I_0  ,\\; H(0) = H_0 ,\\; V(0)=V_0 \\\\
    \\end{align}

  Notes
  -----
  \\(D(t)\\): Population death from covid-19, estimated as a ratio of hospitalized population \\(H(t)\\) \n
  \\(u(t)\\): Cumulative impact of societal measures (reduction) on the transmission coefficient \n
  \\(c\\) : Parameter weighting the cost of societal measures relative to the death toll \n
  \\(S(t)\\): Population susceptible to infection \n
  \\(E(t)\\): Exposed population but not yet infectious \n
  \\(A(t)\\): Infected population but asymptomatic \n
  \\(I(t)\\): Infected population and symptomatic \n
  \\(H(t)\\): Hospitalized population \n
  \\(V(t)\\): Vaccinated population that has not been infected \n
  Other constants: See table 2 page 4 of [Perkins and Espana, 2020](https://link.springer.com/content/pdf/10.1007/s11538-020-00795-y.pdf)
  """

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
      x_0=jnp.array([self.S_0, self.E_0, self.I_0, self.N_0]),
      x_T=None,
      T=20,
      bounds=jnp.array([
        # [-jnp.inf, jnp.inf],
        # [-jnp.inf, jnp.inf],
        # [-jnp.inf, jnp.inf],
        # [-jnp.inf, jnp.inf],
        [0., 2000.],  # Chosen by observation
        [0., 250.],  # "
        [0., 250.],  # "
        [0., 3000.],  # "
        [0., 1.],
      ]),
      terminal_cost=False,
    )

  def dynamics(self, y_t: jnp.ndarray, u_t: float, t: float = None) -> jnp.ndarray:
    S, E, I, N = y_t

    s_dot = jnp.squeeze(self.b*N - self.d*S - self.c*S*I - u_t*S)
    e_dot = jnp.squeeze(self.c*S*I - (self.e+self.d)*E)
    i_dot = jnp.squeeze(self.e*E - (self.g+self.a+self.d)*I)
    n_dot = jnp.squeeze((self.b-self.d)*N - self.a*I)

    y_t_dot = jnp.array([s_dot, e_dot, i_dot, n_dot])
    return y_t_dot
  
  def cost(self, y_t: jnp.ndarray, u_t: float, t: float = None) -> float:
    return self.A * y_t[2] + u_t ** 2

  # def plot_solution(self, x: jnp.ndarray, u: jnp.ndarray) -> None:
  #   sns.set()
  #   plt.figure(figsize=(12, 2.5))
  #   ts_x = jnp.linspace(0, self.T, x.shape[0])
  #   ts_u = jnp.linspace(0, self.T, u.shape[0])
  #
  #   plt.subplot(151)
  #   plt.title('applied control')
  #   plt.plot(ts_u, u)
  #   plt.ylim(-0.1, 1.01)
  #
  #   for idx, title in enumerate(['S', 'E', 'I', 'N']):
  #     plt.subplot(1, 5, idx+2)
  #     plt.title(title)
  #     plt.plot(ts_x, x[:, idx])
  #
  #   plt.tight_layout()
  #   plt.show()
