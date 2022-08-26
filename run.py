# (c) 2021 Nikolaus Howe
import numpy as np
import random

from jax.config import config

from myriad.experiments.e2e_sysid import run_endtoend
from myriad.experiments.mle_sysid import run_mle_sysid
from myriad.experiments.node_e2e_sysid import run_node_endtoend
from myriad.experiments.node_mle_sysid import run_node_mle_sysid
from myriad.useful_scripts import run_setup, run_trajectory_opt, load_node_and_plan
from myriad.probing_numerical_instability import probe, special_probe
from myriad.utils import integrate_time_independent, yield_minibatches

config.update("jax_enable_x64", True)

run_buddy = False


def main():
  #########
  # Setup #
  #########
  hp, cfg = run_setup()
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
  run_trajectory_opt(hp, cfg, save_as='traj_opt_example.pdf')

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


if __name__ == '__main__':
  main()
