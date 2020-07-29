from functools import wraps
import time

import jax
from jax import grad, jacrev, jit, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize

from .config import Config, HParams, SolutionType
from .systems import get_system
from .utils import integrate_fwd


def experiment(hp: HParams, cfg: Config) -> np.ndarray:
  jax.config.update("jax_enable_x64", True)
  if hp.solution == SolutionType.COLLOCATION:
    return collocation_experiment(hp, cfg)
  else:
    raise KeyError


def collocation_experiment(hp: HParams, cfg: Config) -> np.ndarray:
  system = get_system(hp)
  
  # Trapezoidal collocation parameters
  N = hp.collocation_segments # Segments
  h = system.T / N # Segment length

  # Initial collocation guess
  u_guess = np.zeros((N+1,1)) + system.bounds[-1].mean()
  if system.x_T is not None:
    x_guess = np.linspace(system.x_0, system.x_T, num=N+1)
  else:
    x_guess = integrate_fwd(system.dynamics, system.x_0, u_guess, h, N)
  guess, unravel = ravel_pytree(np.hstack((x_guess, u_guess)))

  # Shared decorator for objective and constraint
  def vmapreduce(reduce):
    def _vmapreduce(fn):
      @wraps(fn)
      def wrapper(variables: np.ndarray):
        xu = unravel(variables)
        x, u = xu[:,:-1], xu[:,-1:]
        return reduce(vmap(fn)(x[:-1], x[1:], u[:-1], u[1:]))
      return wrapper
    return _vmapreduce

  # Objective (Eq. 6.4)
  @vmapreduce(np.sum)
  def objective(x_t1: np.ndarray, x_t2: np.ndarray, u_t1: np.float64, u_t2: np.float64) -> np.float64:
    return (h/2) * (system.cost(x_t1, u_t1) + system.cost(x_t2, u_t2))

  # Collocation Constraints (Eq. 6.6)
  @vmapreduce(np.ravel)
  def defect(x_t1: np.ndarray, x_t2: np.ndarray, u_t1: np.float64, u_t2: np.float64) -> np.ndarray:
    left = (h/2) * (system.dynamics(x_t1, u_t1) + system.dynamics(x_t2, u_t2))
    right = x_t2 - x_t1
    return left - right

  # Path and Boundary Constraints 
  bounds = onp.empty((N+1,*system.bounds.shape))
  bounds[:,:,:] = system.bounds # State and control bounds
  bounds[0,0:-1,:] = np.expand_dims(system.x_0, 1) # Starting state
  if system.x_T is not None:
    bounds[-1,0:-1,:] = np.expand_dims(system.x_T, 1) # Ending state
  # Prepare for scipy.optimize.minimize
  bounds = bounds.reshape((-1,2))
  bounds = np.array(bounds)

  _t1 = time.time()
  solution = minimize(
    fun=jit(objective),
    x0=guess,
    method='SLSQP',
    constraints=[{
      'type': 'eq',
      'fun': jit(defect),
      'jac': jit(jacrev(defect)),
    }],
    bounds=bounds,
    jac=jit(grad(objective)),
    options={
      'maxiter': hp.slsqp_maxiter,
      'disp': cfg.verbose
    },
  )
  _t2 = time.time()
  print(f'Solved in {_t2 - _t1} seconds.')

  x = unravel(solution.x)
  if cfg.plot_results:
    system.plot_solution(x)

  return x
