# (c) 2021 Nikolaus Howe

import jax.numpy as jnp
import numpy as np
import random

from absl import app
from jax.config import config

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

from absl import app
from jax.config import config

from myriad.experiments.e2e_sysid import run_endtoend
from myriad.experiments.mle_sysid import run_mle_sysid
from myriad.experiments.node_e2e_sysid import run_node_endtoend
from myriad.experiments.node_mle_sysid import run_node_mle_sysid
from myriad.useful_scripts import run_setup, run_trajectory_opt, load_node_and_plan
from myriad.probing_numerical_instability import probe, special_probe
from myriad.utils import integrate_time_independent, yield_minibatches, get_state_trajectory_and_cost, get_defect
from myriad.trajectory_optimizers.unconstrained_shooting import UnconstrainedShootingOptimizer
from myriad.plotting import plot

config.update("jax_enable_x64", True)

run_buddy = False


def main(argv):
  #########
  # Setup #
  #########
  hp, cfg = run_setup(argv)
  random.seed(hp.seed)
  np.random.seed(hp.seed)

  if run_buddy:
    # random.seed(hp.seed)
    # np.random.seed(hp.seed)
    import experiment_buddy
    experiment_buddy.register(hp.__dict__)
    # tensorboard = experiment_buddy.deploy(host='mila', sweep_yaml="sweep.yaml")
    tensorboard = experiment_buddy.deploy(host='mila', sweep_yaml="")
    # tensorboard = experiment_buddy.deploy(host='', sweep_yaml='')

  ########################################
  # Probing Systems' Numerical Stability #
  ########################################
  # for st in SystemType:
  #   if st in [SystemType.SIMPLECASE, SystemType.INVASIVEPLANT]:
  #     continue
  #   print("system", st)
  #   hp.system = st
  #   probe(hp, cfg)

  # probe(hp, cfg)
  # special_probe(hp, cfg)

  ###########################################
  # Trajectory optimization with true model #
  ###########################################
  # run_trajectory_opt(hp, cfg)

  ######################
  # MLE model learning #
  ######################
  # Parametric, MLE
  # run_mle_sysid(hp, cfg)

  # NODE, MLE
  # run_node_mle_sysid(hp, cfg)

  #############################
  # End to end model learning #
  #############################
  # Parametric, end-to-end
  # run_endtoend(hp, cfg)

  # NODE, end-to-end
  # run_node_endtoend(hp, cfg)

  ###############
  # Noise study #
  ###############
  # study_noise(hp, cfg, experiment_string='mle_sysid')
  # study_noise(hp, cfg, experiment_string='node_mle_sysid')

  ##################
  # Dynamics study #
  ##################
  # study_vector_field(hp, cfg, 'mle', 0)
  # study_vector_field(hp, cfg, 'e2e', 0, file_extension='pdf')

  system = hp.system()
  optimizer = UnconstrainedShootingOptimizer(hp, cfg, system)
  solution = optimizer.unconstrained_solve()
  # print("solution", solution)
  u = jnp.array(solution.x)
  # x = solution['x']
  # u = solution['u']

  true_system = hp.system()
  opt_x, c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, u)
  opt_x = opt_x.squeeze()
  # defect = get_defect(true_system, opt_x)

  plt.subplot(2, 1, 1)
  plt.plot(opt_x)
  plt.subplot(2, 1, 2)
  plt.plot(u)
  print("cost", c)
  plt.show()

  # plot(hp, true_system,
  #      data={'x': opt_x, 'u': u, 'cost': c},
  #      labels={'x': '', 'u': ''},
  #      styles={'x': '-', 'u': '-'},
  #      widths={'x': 2, 'u': 2},
  #      save_as=None)


if __name__ == '__main__':
  app.run(main)
