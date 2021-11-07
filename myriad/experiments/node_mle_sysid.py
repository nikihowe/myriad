# (c) Nikolaus Howe 2021
from __future__ import annotations

import csv
import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl

from pathlib import Path

from myriad.config import Config, HParams, IntegrationMethod
from myriad.neural_ode.create_node import NeuralODE
from myriad.neural_ode.node_training import train
from myriad.trajectory_optimizers import get_optimizer
from myriad.plotting import plot, plot_losses
from myriad.systems.node_system import NodeSystem
from myriad.utils import get_state_trajectory_and_cost, integrate_time_independent, sample_x_init


def run_node_mle_sysid(hp: HParams, cfg: Config) -> None:
  # Instantiate the neural ode. We'll keep updating its parameters
  # (either by training or by loading from save)
  node = NeuralODE(hp, cfg)
  true_opt = get_optimizer(hp, cfg, node.system)
  true_solution = true_opt.solve()
  learned_system = NodeSystem(node, node.system)
  learned_opt = get_optimizer(hp, cfg, learned_system)

  official_trained_for = 0
  for experiment_number in range(0, hp.num_experiments):
    print(f"### EXPERIMENT {experiment_number} ###")
    official_trained_for += hp.num_epochs
    actual_trained_for = official_trained_for  # overwrite if not exact (if we have the info)

    losses_path = f'losses/{hp.system.name}/node_mle_sysid/'
    Path(losses_path).mkdir(parents=True, exist_ok=True)
    losses_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                  f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}.l'

    params_path = f'params/{hp.system.name}/node_mle_sysid/'
    Path(params_path).mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't already exist
    params_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                  f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}.p'

    plots_path = f'plots/{hp.system.name}/node_mle_sysid/'
    progress_plots_path = f'plots/{hp.system.name}/node_mle_sysid/progress_plots/'
    Path(progress_plots_path).mkdir(parents=True, exist_ok=True)
    plots_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                 f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}'

    data_path = f'datasets/{hp.system.name}/node_mle_sysid/'
    Path(data_path).mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't already exist
    data_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}.d'

    try:
      node.load_params(params_path + params_name)
    except FileNotFoundError as e:
      print("unable to find the params file, so we'll train our"
            "model to learn some, and then save them")

      # If the datasets already exist, then load it.
      # If it doesn't then we augment the dataset
      try:
        node.load_dataset(data_path + data_name)
      except FileNotFoundError as e:
        print("unable to find dataset for this experiment, so we'll make our own")
        # (unless it's the first one, in which case we use the
        #  dataset which is already there)
        if experiment_number > 0:
          print("We will now augment the dataset. Currently, the train data are", node.train_data.shape)
          node.augment_datasets()
          print("After augmenting, the train data are", node.train_data.shape)

        # Save the dataset for the next time
        pkl.dump(node.full_data, open(data_path + data_name, 'wb'))

      # Now, we train on this dataset, until early stopping
      node.key, subkey = jax.random.split(node.key)

      # Perform the training
      end_epoch = train(node, save_as=progress_plots_path + plots_name, extension=cfg.file_extension)

      actual_trained_for = official_trained_for - node.hp.num_epochs + end_epoch
      # TODO: do we care how many epochs it trained for?
      # start_epoch += increment * node.train_size

      # Save the learned parameters
      node.save_params(params_path + params_name)

      # Save the losses for this experiment
      # print("saving train and val losses for experiment", experiment_number)
      with open(losses_path + losses_name, 'w') as f:
        write = csv.writer(f)
        for i, t in enumerate(node.losses['ts']):
          write.writerow([t, node.losses['train_loss'][i], node.losses['validation_loss'][i]])

    if cfg.plot:
      #################
      # Planning plot #
      #################
      learned_solution = learned_opt.solve_with_params(node.params)

      true_x, true_c = get_state_trajectory_and_cost(hp, node.system, node.system.x_0, true_solution['u'])
      if node.system.x_T is not None:
        true_defect = []
        for i, s in enumerate(true_x[-1]):
          if node.system.x_T[i] is not None:
            true_defect.append(s - node.system.x_T[i])
        true_defect = np.array(true_defect)
      else:
        true_defect = None

      learned_x, learned_c = get_state_trajectory_and_cost(hp, node.system, node.system.x_0, learned_solution['u'])
      if node.system.x_T is not None:
        learned_defect = []
        for i, s in enumerate(learned_x[-1]):
          if node.system.x_T[i] is not None:
            learned_defect.append(s - node.system.x_T[i])
        learned_defect = np.array(learned_defect)
      else:
        learned_defect = None

      planning_plot_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                           f'{hp.train_size}_{hp.val_size}_{hp.test_size}_' \
                           f'exp_{experiment_number}_planning.{cfg.file_extension}'

      plot(hp, node.system,
           data={'x': true_x,
                 'other_x': learned_x,
                 'u': true_solution['u'],
                 'other_u': learned_solution['u'],
                 'cost': true_c,
                 'other_cost': learned_c,
                 'defect': true_defect,
                 'other_defect': learned_defect},
           labels={'x': ' (true state from controls planned with true model)',
                   'other_x': ' (true state from controls planned with learned model)',
                   'u': ' (planned with true model)',
                   'other_u': ' (planned with learned model)'},
           styles={'x': '-',
                   'other_x': 'x-',
                   'u': '-',
                   'other_u': 'x-'},
           widths={'x': 3,
                   'other_x': 1,
                   'u': 3,
                   'other_u': 1},
           save_as=plots_path + planning_plot_name,
           figsize=cfg.figsize)

      ###############
      # Losses plot #
      ###############
      print("plotting losses for experiment", experiment_number)
      losses_plot_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                         f'{hp.train_size}_{hp.val_size}_{hp.test_size}_' \
                         f'exp_{experiment_number}_training.{cfg.file_extension}'
      if cfg.plot:
        plot_losses(node.hp, losses_path + losses_name, save_as=plots_path + losses_plot_name)

      ###################
      # Prediction plot #
      ###################
      x_0 = sample_x_init(hp, n_batch=1)[0]  # remove the leading (batch) axis
      print("x0", x_0.shape, x_0)

      us = np.random.uniform(low=node.system.bounds[-1, 0],
                             high=node.system.bounds[-1, 1],
                             size=(hp.num_steps + 1, hp.control_size))
      us = jnp.array(us)

      _, predicted_states1 = integrate_time_independent(
        node.system.dynamics, x_0, us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
      )

      _, predicted_states2 = integrate_time_independent(
        learned_system.dynamics, x_0, us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
      )

      Path(plots_path).mkdir(parents=True, exist_ok=True)
      save_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                  f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}' \
                  f'_prediction.{cfg.file_extension}'

      plot(hp, node.system,
           data={'x': predicted_states1,
                 'other_x': predicted_states2,
                 'u': us},
           labels={'x': ' (true state trajectory)',
                   'other_x': ' (state trajectory predicted by learned model)',
                   'u': ' (chosen uniformly at random)'},
           styles={'x': '-',
                   'other_x': '-x',
                   'u': '-'},
           widths={'x': 3,
                   'other_x': 1,
                   'u': 1},
           save_as=plots_path + save_name,
           figsize=cfg.figsize)
