import random

import jax
import jax.numpy as jnp
import numpy as np
import simple_parsing
import gin

from absl import app, flags

import matplotlib.pyplot as plt

from source.config import Config, HParams
from source.config import SamplingApproach, OptimizerType
from source.optimizers import get_optimizer
from source.systems import get_system
from source.utils import integrate

from source.opt_control_neural_ode import run_net

# Prepare experiment settings   # TODO: Use only 1 parsing technique?
parser = simple_parsing.ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")
parser.add_argument("--gin_bindings", type=str)  # Needed for the parser to work in conjunction with absl.flags

key_dict = HParams.__dict__.copy()
key_dict.update(Config.__dict__)
for key in key_dict.keys():
  if "__" not in key:
    flags.DEFINE_string(key, None,    # Parser arguments need to be accepted by the flags
                        'Backward compatibility with previous parser')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "Lab1.A=1.0").')

FLAGS = flags.FLAGS


def main(unused_argv):
  """Main method.
    Args:
      unused_argv: Arguments (unused).
    """
  jax.config.update("jax_enable_x64", True)

  args = parser.parse_args()
  hp = args.hparams
  cfg = args.config
  print(hp)
  print(cfg)

  # Set our seeds for reproducibility
  random.seed(hp.seed)
  np.random.seed(hp.seed)

  # Load config, then build system
  gin_files = ['./source/gin-configs/default.gin']
  gin_bindings = FLAGS.gin_bindings
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)
  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)

  # -----------------------------------------------------------------------
  ######################
  # Place scripts here #
  ######################

  from hyperopt import hp as h
  from hyperopt import fmin, tpe, pyll, STATUS_OK
  import pprint
  pp = pprint.PrettyPrinter(indent=4, width=100)

  # Parameter search for extragradient
  # define an objective function
  def f(space):
    extra_options = {
      'maxiter': space['maxiter'],
      'eta_x': 10**space['eta_x_exp'],
      'eta_v': 10**space['eta_v_exp'],
      'atol': 10**space['atol_exp']
    }

    if optimizer.require_adj:
      x, u, adj = optimizer.solve(extra_options)
    else:
      x, u = optimizer.solve(extra_options)

    print("xs", x.shape)
    print("us", u.shape)

    xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
    obj = optimizer.objective(xs_and_us)
    vio = jnp.linalg.norm(optimizer.constraints(xs_and_us))
    total_cost = obj + 1000*vio
    # print("total_cost", total_cost)
    return {'loss': total_cost, 'status': STATUS_OK}

  # define a search space
  space = {
      'maxiter': h.choice('maxiter', [i*100 for i in range(1, 21)]),
      'eta_x_exp': h.uniform('eta_x_exp', -7, -3),
      'eta_v_exp': h.uniform('eta_v_exp', -7, -3),
      'atol_exp': h.uniform('atol_exp', -10, -2)
    }

  # pp.pprint(pyll.stochastic.sample(space))

  # minimize the objective over the space
  best = fmin(f, space, algo=tpe.suggest, max_evals=3)

  print(best)
  # -----------------------------------------------------------------------

if __name__=='__main__':
  app.run(main)