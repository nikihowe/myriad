# (c) Nikolaus Howe 2021
from scipy.integrate import odeint

import jax.numpy as jnp
import numpy as np
import sys
import unittest

from run import run_trajectory_opt
from source.config import IntegrationMethod, NLPSolverType, OptimizerType, QuadratureRule, SystemType
from source.custom_types import State, Control, Timestep, States
from source.useful_scripts import run_setup
from source.utils import integrate

hp, cfg = run_setup(sys.argv, gin_path='../source/gin-configs/default.gin')


class BasicTests(unittest.TestCase):
  def test_integrate(self):
    # Perform integration using odeint
    def f(t: Timestep, state: State) -> States:
      return state

    y0 = jnp.array([1.])
    t = [0., 1.]
    result_odeint = odeint(f, y0, t, tfirst=True)

    # Perform integration using our 'integrate'
    N = 100
    t = jnp.linspace(0., 1., N)
    h = t[1]

    def f_wrapper(state: State, control: Control, time: Timestep) -> States:
      del control
      return f(time, state)

    _, found_states = integrate(f_wrapper, y0, t, h, N - 1, t, integration_method=IntegrationMethod.RK4)

    # Check that we get similar enough results
    np.testing.assert_almost_equal(result_odeint[-1], found_states[-1], decimal=6,
                                   err_msg=f'our integrator gave {result_odeint[-1]}, '
                                           f'but it should have given {found_states[-1]}',
                                   verbose=True)


class OptimizerTests(unittest.TestCase):
  def test_single_shooting(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.max_iter = 1000
    hp.intervals = 1
    hp.controls_per_interval = 50

    run_trajectory_opt(hp, cfg)

  def test_multiple_shooting(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.max_iter = 1000
    hp.intervals = 20
    hp.controls_per_interval = 3

    run_trajectory_opt(hp, cfg)

  def test_dense_multiple_shooting(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_trapezoidal_collocation(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.COLLOCATION
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_hermite_simpson_collocation(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.COLLOCATION
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.RK4
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)


class IntegrationMethodTests(unittest.TestCase):
  def test_euler(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.EULER
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_heun(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_midpoint(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.MIDPOINT
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_RK4(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.RK4
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)


class QuadratureRuleTests(unittest.TestCase):
  def test_trapozoidal(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.COLLOCATION
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_hermite_simpson(self):
    global hp, cfg

    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.COLLOCATION
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.RK4
    hp.quadrature_rule = QuadratureRule.HERMITE_SIMPSON
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)


class NLPSolverTests(unittest.TestCase):
  def test_ipopt(self):
    hp.seed = 42
    hp.system = SystemType.PENDULUM
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.IPOPT
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_slsqp(self):
    hp.seed = 42
    hp.system = SystemType.PENDULUM
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.SLSQP
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_trust_constr(self):
    hp.seed = 42
    hp.system = SystemType.PENDULUM
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.TRUST
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)

  def test_extragradient(self):
    hp.seed = 42
    hp.system = SystemType.SIMPLECASE
    hp.optimizer = OptimizerType.SHOOTING
    hp.nlpsolver = NLPSolverType.EXTRAGRADIENT
    hp.integration_method = IntegrationMethod.HEUN
    hp.quadrature_rule = QuadratureRule.TRAPEZOIDAL
    hp.max_iter = 1000
    hp.intervals = 50
    hp.controls_per_interval = 1

    run_trajectory_opt(hp, cfg)




if __name__ == '__main__':
  unittest.main()
