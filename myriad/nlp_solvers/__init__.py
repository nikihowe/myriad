# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp
import time

from cyipopt import minimize_ipopt
from scipy.optimize import minimize
from typing import Dict

from myriad.config import Config, HParams, NLPSolverType
from myriad.defaults import learning_rates
from myriad.utils import get_state_trajectory_and_cost

### Import your new nlp solver here ###
from myriad.nlp_solvers.extra_gradient import extra_gradient


def solve(hp: HParams, cfg: Config, opt_dict: Dict) -> Dict[str, jnp.ndarray]:
  """
  Use a the solver indicated in the hyper-parameters to solve the constrained optimization problem.
  Args:
    hp: the hyperparameters
    cfg: the extra hyperparameters
    opt_dict: everything needed for the solve

  Returns
    A dictionary with the optimal controls and corresponding states
    (and for quadratic interpolation schemes, the midpoints too)
  """
  _t1 = time.time()
  opt_inputs = {
    'fun': jax.jit(opt_dict['objective']) if cfg.jit else opt_dict['objective'],
    'x0': opt_dict['guess'],
    'constraints': ({
      'type': 'eq',
      'fun': jax.jit(opt_dict['constraints']) if cfg.jit else opt_dict['constraints'],
      'jac': jax.jit(jax.jacrev(opt_dict['constraints'])) if cfg.jit else jax.jacrev(opt_dict['constraints']),
    }),
    'bounds': opt_dict['bounds'],
    'jac': jax.jit(jax.grad(opt_dict['objective'])) if cfg.jit else jax.grad(opt_dict['objective']),
    'options': {"maxiter": hp.max_iter}
  }

  ### Add new nlp solvers to this list ###
  if hp.nlpsolver == NLPSolverType.EXTRAGRADIENT:
    opt_inputs['method'] = 'exgd'
    if hp.system in learning_rates:
      opt_inputs['options'] = {**opt_inputs['options'], **learning_rates[hp.system]}
    solution = extra_gradient(**opt_inputs)
  elif hp.nlpsolver == NLPSolverType.SLSQP:
    opt_inputs['method'] = 'SLSQP'
    solution = minimize(**opt_inputs)
  elif hp.nlpsolver == NLPSolverType.TRUST:
    opt_inputs['method'] = 'trust-constr'
    solution = minimize(**opt_inputs)
  elif hp.nlpsolver == NLPSolverType.IPOPT:
    opt_inputs['method'] = 'ipopt'
    solution = minimize_ipopt(**opt_inputs)
  else:
    print("Unknown NLP solver. Please choose among", list(NLPSolverType.__members__.keys()))
    raise ValueError
  _t2 = time.time()
  if cfg.verbose:
    print('Solver exited with success:', solution['success'])
    print(f'Completed in {_t2 - _t1} seconds.')

    system = hp.system()
    opt_x, c = get_state_trajectory_and_cost(hp, system, system.x_0, (opt_dict['unravel'](solution['x']))[1])
    print('Cost given by solver:', solution['fun'])
    print("Cost given by integrating the control trajectory:", c)
    if system.x_T is not None:
      achieved_last_state = opt_x[-1]
      desired_last_state = system.x_T
      defect = []
      for i, el in enumerate(desired_last_state):
        if el is not None:
          defect.append(achieved_last_state[i] - el)
      print("Defect:", defect)

  lmbda = None
  if hp.nlpsolver == NLPSolverType.IPOPT:
    lmbda = solution.info['mult_g']
  elif hp.nlpsolver == NLPSolverType.TRUST:
    lmbda = solution['v']
  elif hp.nlpsolver == NLPSolverType.EXTRAGRADIENT:
    lmbda = solution['v']
  # print("the full solution was", solution)
  # raise SystemExit

  results = {'x': (opt_dict['unravel'](solution['x']))[0],
             'u': (opt_dict['unravel'](solution['x']))[1],
             'xs_and_us': solution['x'],
             'cost': solution['fun']}

  if lmbda is not None:
    results['lambda'] = lmbda

  return results
