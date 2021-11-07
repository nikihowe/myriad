# (c) Nikolaus Howe 2021
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import typing

if typing.TYPE_CHECKING:
  from myriad.neural_ode.create_node import NeuralODE

from jax.flatten_util import ravel_pytree
from tqdm import trange
from typing import Callable, Optional, Tuple

from myriad.custom_types import Batch, Controls, Cost, Epoch
from myriad.useful_scripts import plan_and_plot, plot_trajectory
from myriad.utils import integrate_time_independent, integrate_time_independent_in_parallel, plan_with_node_model, \
  yield_minibatches


# Perform node.hp.num_epochs of minibatched gradient descent.
# Store losses in the "losses" dict. Return the termination epoch.
def train(node: NeuralODE,
          start_epoch: Epoch = 0,
          also_record_planning_loss: bool = False,
          save_as: Optional[str] = None,
          extension: Optional[str] = 'png') -> Epoch:
  @jax.jit
  def loss(params: hk.Params, minibatch: Batch) -> Cost:
    # assert jnp.isfinite(minibatch)  # had to comment out because of jitting

    def apply_net(x, u):
      net_input = jnp.append(x, u)
      # print("net input", net_input)
      return node.net.apply(params, net_input)

    # Extract controls and true state trajectory
    controls = minibatch[:, :, node.hp.state_size:]
    true_states = minibatch[:, :, :node.hp.state_size]

    # Extract starting states
    # print("true states", true_states.shape)
    start_states = true_states[:, 0, :]

    # Use neural net to predict state trajectory
    _, predicted_states = integrate_time_independent_in_parallel(apply_net, start_states,
                                                                 controls, node.stepsize, node.num_steps,
                                                                 node.hp.integration_method)

    return jnp.mean((predicted_states - true_states) * (predicted_states - true_states))  # MSE

  # Gradient descent on the loss function in scope
  @jax.jit
  def update(params: hk.Params, opt_state: optax.OptState, minibatch: Batch) -> Tuple[hk.Params, optax.OptState]:
    grads = jax.grad(loss)(params, minibatch)
    updates, opt_state = node.opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  best_val_loss = 10e10
  best_params = None
  epoch = None
  count = 0
  print("check loss frequency is", node.hp.loss_recording_frequency)
  # train_loss, validation_loss = calculate_losses(node, loss, 0, also_record_planning_loss)
  for epoch in trange(node.hp.num_epochs):
    overall_epoch = start_epoch + epoch * node.hp.train_size
    if epoch % node.hp.loss_recording_frequency == 0:
      # As a side-effect, this function also fills the loss lists
      # print('calculating losses')
      train_loss, validation_loss = calculate_losses(node, loss, overall_epoch, also_record_planning_loss)
      print(epoch, train_loss, validation_loss)

    # Plot progress so far
    if epoch % node.hp.plot_progress_frequency == 0:
      if node.cfg.plot and save_as is not None:
        print("saving progress plot :)")
        plot_progress(node, overall_epoch, save_as, extension)

    # Early stopping
    if epoch % node.hp.early_stop_check_frequency == 0:
      if count >= node.hp.early_stop_threshold:
        print(f"Stopping early at epoch {epoch}. Threshold was {node.hp.early_stop_threshold} epochs.")
        break

      # Update early stopping counts/values
      if validation_loss >= best_val_loss:
        count += node.hp.early_stop_check_frequency
      else:
        best_val_loss = validation_loss
        best_params = hk.data_structures.to_immutable_dict(node.params)
        count = 0

    # Descend on entire dataset, in minibatches
    # NOTE: when we add new data to the train set, we still only use the same number of
    #       minibatches to count as an "epoch" (so after the first experiment, when
    #       we complete an epoch, there will still be unseen data each time)
    for mb in yield_minibatches(node.hp, node.hp.train_size, node.train_data):
      node.params, node.opt_state = update(node.params, node.opt_state, mb)

  # Save the best params
  node.params = best_params

  if epoch and node.cfg.verbose:
    print("Trained for {} epochs on dataset of size {}".format(epoch, node.hp.train_size))

  return epoch


def calculate_losses(node: NeuralODE,
                     loss_fn: Callable[[hk.Params, Batch], float],
                     overall_epoch: int,
                     also_record_planning_losses: bool = False) -> Tuple[Cost, Cost]:
  # Record how many training points we've used
  node.losses['ts'].append(overall_epoch)

  # Calculate losses
  cur_loss = loss_fn(node.params, next(yield_minibatches(node.hp, node.hp.train_size, node.train_data)))
  node.losses['train_loss'].append(cur_loss)
  validation_loss = loss_fn(node.params, next(yield_minibatches(node.hp, node.hp.val_size, node.validation_data)))
  node.losses['validation_loss'].append(validation_loss)
  node.losses['loss_on_opt'].append(loss_fn(node.params, node.true_x_and_u_opt[jnp.newaxis]))

  if also_record_planning_losses:
    planning_loss, planning_defect, u = calculate_planning_loss(node)
    node.losses['control_costs'].append(planning_loss)
    if planning_defect is not None:
      node.losses['constraint_violation'].append(planning_defect)

    # Calculate divergences from the optimal trajectories
    node.losses['divergence_from_optimal_us'].append(divergence_from_optimal_us(node, u))
    node.losses['divergence_from_optimal_xs'].append(divergence_from_optimal_xs(node, u))

  return cur_loss, validation_loss


def calculate_planning_loss(node: NeuralODE) -> Tuple[Cost, Optional[Cost], Controls]:
  # Get the optimal controls, and cost of applying them
  _, u = plan_with_node_model(node)
  _, xs = integrate_time_independent(node.system.dynamics, node.system.x_0, u, node.stepsize,  # true dynamics
                                     node.num_steps, node.hp.integration_method)

  # We only want the states at boundaries of shooting intervals
  xs_interval_start = xs[::node.hp.controls_per_interval]
  xs_and_us, unused_unravel = ravel_pytree((xs_interval_start, u))
  cost1 = node.optimizer.objective(xs_and_us)

  # Calculate the final constraint violation, if present
  if node.system.x_T is not None:
    cv = node.system.x_T - xs[-1]
    # if node.cfg.verbose:
    # print("constraint violation", cv)
    cost2 = jnp.linalg.norm(cv)
  else:
    cost2 = None

  return cost1, cost2, u


# TODO: jit
# This is the "outer" loss of the problem, one of the main things we care about.
# Another "outer" loss, which gives a more RL flavour,
# is the integral cost of applying controls in the true dynamics,
# and the final constraint violation (if present) when controls in the true dynamics.
def divergence_from_optimal_us(node: NeuralODE, us: Controls) -> Cost:
  assert len(us) == len(node.true_opt_us)
  return jnp.mean((us - node.true_opt_us) * (us - node.true_opt_us))  # MS


def divergence_from_optimal_xs(node: NeuralODE, us: Controls) -> Cost:
  # Get true state trajectory from applying "optimal" controls
  _, xs = integrate_time_independent(node.system.dynamics, node.system.x_0, us,
                                     node.stepsize, node.num_steps, node.hp.integration_method)

  assert len(xs) == len(node.true_opt_xs)
  return jnp.mean((xs - node.true_opt_xs) * (xs - node.true_opt_xs))  # MSE


def plot_progress(node, trained_for, save_as, extension, also_plan=False):
  plot_trajectory(node,
                  optimal=True,
                  title="Prediction on optimal trajectory after {} epochs".format(trained_for),
                  save_as=save_as + str(trained_for) + f"_im_opt.{extension}")
  plot_trajectory(node,
                  optimal=False,
                  title="Prediction on train trajectory after {} epochs".format(trained_for),
                  save_as=save_as + str(trained_for) + f"_im_train_rand.{extension}")
  plot_trajectory(node,
                  optimal=False,
                  validation=True,
                  title="Prediction on validation trajectory after {} epochs".format(trained_for),
                  save_as=save_as + str(trained_for) + f"_im_val_rand.{extension}")
  # Use the network for planning
  if also_plan:
    plan_and_plot(node,
                  title="Planning after {} epochs".format(trained_for),
                  save_as=save_as + str(trained_for) + f"_plan.{extension}")
