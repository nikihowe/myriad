import random

import jax
import jax.numpy as jnp
import numpy as np
import simple_parsing
import gin

from absl import app, flags

import matplotlib.pyplot as plt

from source.config import Config, HParams
from source.config import SystemType, SamplingApproach, OptimizerType, NLPSolverType
from source.optimizers import get_optimizer
from source.systems import get_system
from source.utils import integrate
from source.plotting import plot

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

  ########################################################################################################3

  # -----------------------------------------------------------------------
  ######################
  # Place scripts here #
  ######################
   
  # put this in run.py
  # optimizer = get_optimizer(hp, cfg, system)
  # if optimizer.require_adj:
  #   x, u, adj = optimizer.solve()
  # else:
  #   x, u = optimizer.solve()
  #
  # num_steps = hp.intervals*hp.controls_per_interval
  # stepsize = system.T / num_steps
  # _, opt_x = integrate(system.dynamics, system.x_0, u,
  #                     stepsize, num_steps, None, hp.order)
  #
  # if cfg.plot_results:
  #   if optimizer.require_adj:
  #     plot(system,
  #          data={'x': x, 'u': u, 'adj': adj, 'other_x': opt_x},
  #          labels={'x': ' (from solver)',
  #                  'u': 'Controls from solver',
  #                  'adj': 'Adjoint from solver',
  #                  'other_x': ' (from integrating controls from solver)'})
  #   else:
  #     plot(system,
  #          data={'x': x, 'u': u, 'other_x': opt_x},
  #          labels={'x': ' (from solver)',
  #                  'u': 'Controls from solver',
  #                  'other_x': ' (from integrating controls from solver)'})
  #
  # xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
  # if hp.optimizer != OptimizerType.FBSM:
  #   print("control cost", optimizer.objective(xs_and_us))
  #   print('constraint_violations', jnp.linalg.norm(optimizer.constraints(xs_and_us)))
  # raise SystemExit

  # Can choose which sampling approach to use, for different effects
  # (could even put together multiple sampling approaches in each plot)
  sas = [SamplingApproach.FIXED, SamplingApproach.PLANNING]
  sas = [SamplingApproach.FIXED]
  nlp = [NLPSolverType.IPOPT, NLPSolverType.TRUST]
  exp = [SystemType.CANCER, SystemType.SIMPLECASE, SystemType.MOLDFUNGICIDE]
  exp = [SystemType.SIMPLECASE, SystemType.MOLDFUNGICIDE]
  for e in exp:
    for s in sas:
      for n in nlp:
        hp.system = e
        hp.sampling_approach = s
        hp.nlpsolver = n
        key = jax.random.PRNGKey(42)
        losses = run_net(key, hp, cfg,
                         train_size=1_000,
                         increment=1,
                         num_experiments=6)

  # -----------------------------------------------------------------------

if __name__=='__main__':
  app.run(main)