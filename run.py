import random

import jax
import numpy as np
import simple_parsing
import gin

from absl import app, flags

import matplotlib.pyplot as plt

from source.config import Config, HParams
from source.config import SamplingApproach, OptimizerType
from source.optimizers import get_optimizer
from source.systems import get_system

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

  # --------------------------------------------

  # Run trajectory optimization
  # optimizer = get_optimizer(hp, cfg, system)
  # if optimizer.require_adj:
  #     x, u, adj = optimizer.solve()

  #     if cfg.plot_results:
  #       system.plot_solution(x, u, adj)
  # else:
  #     x, u = optimizer.solve()

  #     if cfg.plot_results:
  #         system.plot_solution(x, u)

  # raise SystemExit

  # --------------------------------------------

  # Run neural network
  # name = "source/params/{}_{}_{}.p".format(hp.system.name, n, date_string)

  losses_simple = run_net(hp, cfg, sampling_approach=SamplingApproach.UNIFORM)
  losses_informed = run_net(hp, cfg, sampling_approach=SamplingApproach.PLANNING)

  ts = [(i+1)*1000 for i in range(0, len(losses_simple['train_loss']))]

  plt.figure(figsize=(10, 9))

  ax = plt.subplot(2, 2, 1)
  plt.plot(ts, losses_simple['train_loss'], ".-", label="train")
  plt.plot(ts, losses_simple['validation_loss'], ".-", label="validation")
  plt.title("\"simple\" approach's loss over time")
  plt.yscale('log')
  ax.set_ylim([0,1000])
  ax.legend()
  
  ax = plt.subplot(2, 2, 2)
  plt.plot(ts, losses_informed['train_loss'], ".-", label="train")
  plt.plot(ts, losses_informed['validation_loss'], ".-", label="validation")
  plt.title("\"informed\" approach's loss over time")
  plt.yscale('log')
  ax.set_ylim([0,1000])
  ax.legend()

  # ax = plt.subplot(2, 2, 3)
  # plt.plot(losses_simple['loss_on_opt'], "o-", label="simple")
  # plt.plot(losses_informed['loss_on_opt'], "o-", label="informed")
  # plt.title("loss over time on true optimal trajectory")
  # plt.yscale('log')
  # ax.legend()

  ax = plt.subplot(2, 2, 3)
  plt.plot(ts, losses_simple['control_costs'], ".-", label="simple")
  plt.plot(ts, losses_informed['control_costs'], ".-", label="informed")
  plt.title("cost of applying \"optimal\" controls")
  # plt.yscale('log')
  ax.legend()

  ax = plt.subplot(2, 2, 4)
  plt.plot(ts, losses_simple['constraint_violations'], ".-", label="simple")
  plt.plot(ts, losses_informed['constraint_violations'], ".-", label="informed")
  plt.title("constraint violation when applying those controls")
  # plt.yscale('log')
  ax.legend()

  plt.show()

  # date_string = date.today().strftime("%Y-%m-%d")

  # # Train for different amounts of time
  # for n in [i*10_000 for i in range(1, 11)]:
  #   print("num_training_steps", n)
  #   run_net(hp, cfg, num_training_steps=n,
  #           save_plot_title="{}_{}_{}".format(hp.system.name, date_string, n))

  # Test the quality of the different trainings
  # from datetime import date
  # for n in [i*10_000 for i in range(1, 11)]:
  #   date_string = date.today().strftime("%Y-%m-%d")
  #   name = "source/params/{}_{}_{}.p".format(hp.system.name, n, date_string)
  #   run_net(hp, cfg, use_params=name,
  #           save_plot_title="{}_{}".format(hp.system.name, n))
  #   break

if __name__=='__main__':
  app.run(main)