# (c) 2021 Nikolaus Howe
from __future__ import annotations

import csv
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle as pkl

from pathlib import Path
from typing import Dict, Tuple, Union

from myriad.config import Config, HParams, IntegrationMethod, SystemType
from myriad.defaults import param_guesses
from myriad.trajectory_optimizers import get_optimizer
from myriad.plotting import plot, plot_losses
from myriad.utils import integrate_time_independent, integrate_time_independent_in_parallel, \
  get_state_trajectory_and_cost, get_defect, sample_x_init, generate_dataset


def run_mle_sysid(hp: HParams, cfg: Config) -> None:
  if hp.system not in param_guesses:
    print("We do not currently support that kind of system for sysid. Exiting...")
    return

  test_system = hp.system()

  # Create, or load, a train and validation (and test, unused) set.
  dataset_size = hp.train_size + hp.val_size + hp.test_size
  file_path = f'datasets/{hp.system.name}/mle_sysid/'
  Path(file_path).mkdir(parents=True, exist_ok=True)
  file_name = f'noise_{hp.noise_level}_{hp.train_size}_{hp.val_size}_{hp.test_size}.d'
  try:
    dataset = pkl.load(open(file_path + file_name, 'rb'))
    dataset = jnp.array(dataset)
    print("loaded the dataset from file")
  except FileNotFoundError as e:
    print("unable to find the file, so we're making our own")
    dataset = generate_dataset(hp, cfg)
    pkl.dump(dataset, open(file_path + file_name, 'wb'))

  assert dataset.shape == (dataset_size, hp.num_steps + 1, hp.state_size + hp.control_size)

  if cfg.verbose:
    print("full dataset", dataset.shape)

  assert np.isfinite(dataset).all()

  # Perform the learning
  train_set, val_set, test_set = dataset[:hp.train_size], dataset[hp.train_size:-hp.test_size], dataset[-hp.test_size:]
  if cfg.verbose:
    print("train set", train_set.shape)
    print("val set", val_set.shape)
    print("test set", test_set.shape)

  losses_path = f'losses/{hp.system.name}/mle_sysid/'
  Path(losses_path).mkdir(parents=True, exist_ok=True)
  losses_name = f'noise_{hp.noise_level}_{hp.train_size}_{hp.val_size}_{hp.test_size}.l'

  params_path = f'params/{hp.system.name}/mle_sysid/'
  Path(params_path).mkdir(parents=True, exist_ok=True)
  params_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_{hp.train_size}_{hp.val_size}_{hp.test_size}.p'

  plots_path = f'plots/{hp.system.name}/mle_sysid/'
  Path(plots_path).mkdir(parents=True, exist_ok=True)
  plots_name = f'noise_{hp.noise_level}_{hp.train_size}_{hp.val_size}_{hp.test_size}'

  try:
    if cfg.load_params_if_saved:
      params = pkl.load(open(params_path + params_name, 'rb'))
      print("loaded params from file")
    else:
      raise FileNotFoundError
  except FileNotFoundError as e:
    print("unable to find the params file, so we'll train "
          "our model to learn some, and then save them")

    # Make an initial guess for the system parameters
    params = param_guesses[hp.system]

    # Initialize optimizer
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # Calculate the MSE between the simulated trajectory and the real one
    @jax.jit
    def loss(given_params, dataset, epoch):
      # print("the given params are", given_params)
      def dynamics(x, u):
        return test_system.parametrized_dynamics(given_params, x, u)

      train_xs = dataset[:, :, :hp.state_size]
      train_us = dataset[:, :, hp.state_size:]
      start_xs = train_xs[:, 0, :]
      # if cfg.verbose:
      #   print("train xs", train_xs.shape)
      #   print("train us", train_us.shape)
      #   print("start train xs", start_xs.shape)

      _, predicted_states = integrate_time_independent_in_parallel(
        dynamics, start_xs, train_us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
      )
      # assert jnp.isfinite(predicted_states).all()

      # if cfg.verbose:
      #   print("the predicted states are", predicted_states.shape)
      # print(predicted_states)

      # Calculate the loss, using a discount factor
      # to incentivize learning the earlier part of the
      # trajectory first. This seems to avoid local minima.
      # print('predicted', predicted_states.shape)
      # print('true', train_xs.shape)
      diff = predicted_states - train_xs
      sq_diff = diff * diff
      long = jnp.mean(sq_diff, axis=(0, 2))  # average all axes except time
      discount = (1 - 1 / (1 + jnp.exp(2 + 0.000001 * epoch))) ** jnp.arange(len(long))
      if hp.system in [SystemType.BACTERIA, SystemType.MOUNTAINCAR, SystemType.CARTPOLE]:
        print("min discount", discount[-1])
      else:
        discount = 1.
      return jnp.mean(long * discount)

    # Gradient descent on the loss function already in scope
    @jax.jit
    def update(params: Dict[str, Union[float, jnp.ndarray]],
               opt_state: optax.OptState, minibatch: jnp.ndarray,
               epoch: int) \
            -> Tuple[Dict[str, Union[float, jnp.ndarray]], optax.OptState]:
      grads = jax.grad(loss)(params, minibatch, epoch)
      updates, opt_state = opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    # MLE train
    epochs = []
    train_losses = []
    val_losses = []
    best_val_loss = None
    best_params = None
    check_frequency = 500
    count = 0
    for epoch in range(hp.num_epochs * 10):
      if epoch % check_frequency == 0:
        cur_loss = loss(params, train_set, epoch)
        val_loss = loss(params, val_set, epoch)
        epochs.append(epoch)
        train_losses.append(cur_loss)
        val_losses.append(val_loss)
        if cfg.verbose:
          print("loss", cur_loss)
          print("val loss", val_loss)
          if np.isnan(cur_loss):
            print("current params", params)
            print("train set", train_set)
            with open('t_set', 'wb') as afile:
              pkl.dump(train_set, afile)
            raise SystemExit
          # print("params", params)

        # writer.add_scalar('loss/train', cur_loss, epoch)
        # writer.add_scalar('loss/val', val_loss, epoch)

        # Break if we have converged
        if best_val_loss is None or val_loss < best_val_loss:
          best_val_loss = val_loss
          best_params = params
          count = 0
        else:
          if count > hp.early_stop_threshold:
            print("stopping early at epoch", epoch)
            break

        # If we're still going, increase the count
        count += check_frequency

        if epoch % 2500 == 0:
          # Plot the situation
          first_xs = train_set[0, :, :hp.state_size]
          first_us = train_set[0, :, hp.state_size:]

          @jax.jit
          def dynamics(x, u):
            return test_system.parametrized_dynamics(params, x, u)

          _, predicted_states = integrate_time_independent(
            dynamics, first_xs[0], first_us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
          )
          # if cfg.verbose:
          #   print("plotting xs", first_xs.shape)
          #   print("plotting us", first_us.shape)

          # Plot states
          plt.subplot(2, 1, 1)
          plt.plot(first_xs, label="true xs")
          plt.plot(predicted_states, label="predicted xs")
          plt.legend()

          # Plot controls
          plt.subplot(2, 1, 2)
          plt.plot(first_us, label="true us")
          plt.legend()

          # Save the plot
          plt.savefig(f"{plots_path + plots_name}_epoch_{epoch}.png")
          plt.close()

      # Update the params
      params, opt_state = update(params, opt_state, train_set, epoch)

    print("saving the final params", best_params)
    pkl.dump(best_params, open(params_path + params_name, 'wb'))

    print("saving the train and val losses")
    with open(losses_path + losses_name, 'w') as f:
      write = csv.writer(f)
      for i, ep in enumerate(epochs):
        write.writerow([ep, train_losses[i], val_losses[i]])

    # Use the best params for the plotting, etc.
    params = best_params

  print("the final params are", params)

  # Now we compare the performance when using the learned model for planning,
  # compared with the performance of using the original model for planning.
  true_system = hp.system()
  learned_system = hp.system(**params)

  # Uncomment the following 12 lines if you want to verify the performance on the
  # dataset that was used for training
  # (note: this should _not_ be used as a form of evaluation!)

  # dataset = pkl.load(open(file_path, 'rb'))
  # dataset = jnp.array(dataset)
  # first_xs = dataset[0, :, :state_size]
  # first_us = dataset[0, :, state_size:]
  # _, predicted_states1 = integrate_time_independent(
  #   p1.dynamics, first_xs[0], first_us, stepsize, num_steps, IntegrationMethod.HEUN
  # )
  # _, predicted_states2 = integrate_time_independent(
  #   p2.dynamics, first_xs[0], first_us, stepsize, num_steps, IntegrationMethod.HEUN
  # )
  # print("first xs", first_xs.shape)
  # print("first us", first_us.shape)

  # Test imitation on random controls and a random start point
  x_0 = sample_x_init(hp, n_batch=1)[0]  # remove the leading (batch) axis
  print("x0", x_0.shape, x_0)

  us = np.random.uniform(low=true_system.bounds[-1, 0],
                         high=true_system.bounds[-1, 1],
                         size=(hp.num_steps + 1, hp.control_size))
  us = jnp.array(us)

  _, predicted_states1 = integrate_time_independent(
    true_system.dynamics, x_0, us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
  )

  _, predicted_states2 = integrate_time_independent(
    learned_system.dynamics, x_0, us, hp.stepsize, hp.num_steps, IntegrationMethod.HEUN
  )

  save_path = f'plots/{hp.system.name}/mle_sysid/'
  Path(save_path).mkdir(parents=True, exist_ok=True)
  save_name = f'{hp.train_size}_{hp.val_size}_' \
              f'noise_{hp.noise_level}_{hp.test_size}_prediction.{cfg.file_extension}'

  plot(hp, true_system,
       data={'x': predicted_states1,
             'other_x': predicted_states2,
             'u': us},
       labels={'x': ' (true state trajectory)',
               'other_x': ' (state trajectory predicted by learned model)',
               'u': ' (chosen uniformly at random)'},
       styles={'x': '-',
               'other_x': 'x-',
               'u': '-'},
       widths={'x': 3,
               'other_x': 1,
               'u': 1},
       save_as=save_path + save_name,
       figsize=cfg.figsize)

  # plt.figure(figsize=(9, 7))
  # plt.plot(predicted_states1, '.', label="true")
  # plt.plot(predicted_states2, label="predicted")
  # plt.title("Imitation")
  # plt.legend()
  # plt.show()
  #
  # Perform optimal control using the learned dynamics and the real dynamics
  true_opt = get_optimizer(hp, cfg, true_system)
  true_solution = true_opt.solve()

  learned_opt = get_optimizer(hp, cfg, learned_system)
  learned_solution = learned_opt.solve()

  true_x, true_c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, true_solution['u'])
  true_defect = get_defect(true_system, true_x)

  learned_x, learned_c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, learned_solution['u'])
  learned_defect = get_defect(true_system, learned_x)

  save_path = f'plots/{hp.system.name}/mle_sysid/'
  Path(save_path).mkdir(parents=True, exist_ok=True)
  save_name = f'noise_{hp.noise_level}_{hp.train_size}_{hp.val_size}_{hp.test_size}_planning.{cfg.file_extension}'

  plot(hp, true_system,
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
       save_as=save_path + save_name,
       figsize=cfg.figsize)

  losses_plot_name = f'noise_{hp.noise_level}_{hp.train_size}_{hp.val_size}_{hp.test_size}' \
                     f'_training.{cfg.file_extension}'
  plot_losses(hp, losses_path + losses_name, save_as=save_path + losses_plot_name)
