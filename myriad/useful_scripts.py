# (c) 2021 Nikolaus Howe
from __future__ import annotations

import gin
import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import simple_parsing

from absl import flags
from jax.flatten_util import ravel_pytree
from jax.config import config
from pathlib import Path
from typing import Tuple

from myriad.config import HParams, Config
from myriad.custom_types import Cost, Defect, Optional
from myriad.neural_ode.create_node import NeuralODE
from myriad.trajectory_optimizers import get_optimizer, TrajectoryOptimizer
from myriad.utils import get_defect, integrate_time_independent, get_state_trajectory_and_cost, plan_with_node_model
from myriad.plotting import plot
from myriad.systems import FiniteHorizonControlSystem
from myriad.systems.node_system import NodeSystem
from myriad.config import OptimizerType

config.update("jax_enable_x64", True)


def run_trajectory_opt(hp: HParams, cfg: Config, save_as: str = None,
                       params_path: str = None) -> Tuple[Cost, Optional[Defect]]:
  plot_path = f'plots/{hp.system.name}/trajectory_opt/'
  Path(plot_path).mkdir(parents=True, exist_ok=True)
  if save_as is not None:
    save_as = plot_path + save_as

  if params_path is not None:
    params = pkl.load(open(params_path, 'rb'))
    system = hp.system(**params)
    print("loaded params:", params)
  else:
    system = hp.system()
    print("made default system")
  optimizer = get_optimizer(hp, cfg, system)
  solution = optimizer.solve()
  x = solution['x']
  u = solution['u']
  if optimizer.require_adj:
    adj = solution['adj']

  true_system = hp.system()
  opt_x, c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, u)
  defect = get_defect(true_system, opt_x)

  if cfg.plot:
    if cfg.pretty_plotting:
      plot(hp, true_system,
           data={'x': opt_x, 'u': u, 'cost': c, 'defect': defect},
           labels={'x': '', 'u': ''},
           styles={'x': '-', 'u': '-'},
           widths={'x': 2, 'u': 2},
           save_as=save_as)
    else:  # We also want to plot the state trajectory we got from the solver
      if optimizer.require_adj:
        plot(hp, true_system,
             data={'x': x, 'u': u, 'adj': adj, 'other_x': opt_x, 'cost': c, 'defect': defect},
             labels={'x': ' (from solver)',
                     'u': 'Controls from solver',
                     'adj': 'Adjoint from solver',
                     'other_x': ' (integrated)'},
             save_as=save_as)
      else:
        plot(hp, true_system,
             data={'x': x, 'u': u, 'other_x': opt_x, 'cost': c, 'defect': defect},
             labels={'x': ' (from solver)',
                     'u': 'Controls from solver',
                     'other_x': ' (from integrating dynamics)'},
             save_as=save_as)

  return c, defect


def run_node_trajectory_opt(hp: HParams, cfg: Config, save_as: str = None,
                            params_path: str = None) -> Tuple[Cost, Optional[Defect]]:
  true_system = hp.system()

  node = NeuralODE(hp, cfg)
  node.load_params(params_path)
  node_system = NodeSystem(node, true_system)
  node_optimizer = get_optimizer(hp, cfg, node_system)

  node_solution = node_optimizer.solve_with_params(node.params)
  u = node_solution['u']

  opt_x, c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, u)
  defect = get_defect(true_system, opt_x)

  if cfg.plot:
    plot(hp, true_system,
         data={'x': opt_x, 'u': u, 'cost': c, 'defect': defect},
         labels={'x': '', 'u': ''},
         styles={'x': '-', 'u': '-'},
         save_as=save_as)

  return c, defect


def run_setup():
  # def run_setup(gin_path='./myriad/gin-configs/default.gin'):  # note: no longer need Gin

  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(HParams, dest="hparams")
  parser.add_arguments(Config, dest="config")
  # parser.add_argument("--gin_bindings", type=str)  # Needed for the parser to work in conjunction with absl.flags

  key_dict = HParams.__dict__.copy()
  key_dict.update(Config.__dict__)
  print("the key dict is", key_dict)
  for key in key_dict.keys():
    if "__" not in key:
      flags.DEFINE_string(key, None,  # Parser arguments need to be accepted by the flags
                          'Backward compatibility with previous parser')

  flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "Lab1.A=1.0").')

  jax.config.update("jax_enable_x64", True)

  args = parser.parse_args()
  hp = args.hparams
  cfg = args.config
  print(hp)
  print(cfg)

  # Set our seeds for reproducibility
  np.random.seed(hp.seed)

  return hp, cfg


def plot_zero_control_dynamics(hp, cfg):
  system = hp.system()
  optimizer = get_optimizer(hp, cfg, system)
  num_steps = hp.intervals * hp.controls_per_interval
  stepsize = system.T / num_steps
  zero_us = jnp.zeros((num_steps + 1,))

  _, opt_x = integrate_time_independent(system.dynamics, system.x_0, zero_us,
                                        stepsize, num_steps, hp.integration_method)

  plot(hp, system,
       data={'x': opt_x, 'u': zero_us},
       labels={'x': 'Integrated state',
               'u': 'Zero controls'})

  xs_and_us, unused_unravel = ravel_pytree((opt_x, zero_us))
  if hp.optimizer != OptimizerType.FBSM:
    print("control cost from optimizer", optimizer.objective(xs_and_us))
    print('constraint violations from optimizer', jnp.linalg.norm(optimizer.constraints(xs_and_us)))


# Plot the given control and state trajectory. Also plot the state
# trajectory which occurs when using the neural net for dynamics.
# If "optimal", do the same things as above but using the true
# optimal controls and corresponding true state trajectory.
# "extra_u" is just a way to plot an extra control trajectory.
def plot_trajectory(node: NeuralODE,
                    optimal: bool = False,
                    x: jnp.ndarray = None,
                    u: jnp.ndarray = None,
                    validation: bool = False,
                    title: str = None,
                    save_as: str = None) -> None:
  if validation:
    dset = node.validation_data
  else:
    dset = node.train_data

  if x is None:
    x: jnp.ndarray = dset[-1, :, :node.hp.state_size]
  if u is None:
    u: jnp.ndarray = dset[-1, :, node.hp.state_size:]

  apply_net = lambda x, u: node.net.apply(node.params, jnp.concatenate((x, u), axis=0))  # use nonlocal net and params

  # if node.cfg.verbose:
  #   print("states to plot", x.shape)
  #   print("controls to plot", u.shape)

  if optimal:
    x = node.true_opt_xs
    u = node.true_opt_us

  # Get states when using those controls
  _, predicted_states = integrate_time_independent(apply_net, x[0], u,
                                                   node.stepsize, node.num_steps,
                                                   node.hp.integration_method)

  # Get the true integrated cost of these controls
  _, control_cost = get_state_trajectory_and_cost(node.hp, node.system, x[0], u)

  # If there is a final state, also report the defect
  defect = None
  if node.system.x_T is not None:
    defect = []
    for i, s in enumerate(predicted_states[-1]):
      if node.system.x_T[i] is not None:
        defect.append(s - node.system.x_T[i])
    defect = np.array(defect)

  # Plot
  plot(hp=node.hp,
       system=node.system,
       data={'x': x, 'u': u, 'other_x': predicted_states, 'cost': control_cost,
             'defect': defect},
       labels={'x': ' (true)', 'u': '', 'other_x': ' (predicted)'},
       title=title, save_as=save_as)


# Plan with the model. Plot the controls from planning and corresponding true state trajectory.
# Compare it with the true optimal controls and corresponding state trajectory.
def plan_and_plot(node: NeuralODE, title: str = None, save_as: str = None) -> None:
  planned_x, planned_us = plan_with_node_model(node)
  xs, cost = get_state_trajectory_and_cost(node.hp, node.system, node.system.x_0, planned_us)

  # If this is the best cost so far, update the best guess for us
  # TODO: I don't think this is the place to do this... where is better?
  if node.best_guess_us_cost is None or cost < node.best_guess_us_cost:
    print("updating best us with a cost of", cost)
    node.best_guess_us = planned_us
    node.best_guess_us_cost = cost
    new_guess, _ = ravel_pytree((planned_x, planned_us))
    node.optimizer.guess = new_guess

  # single_traj_train_controls = node.train_data[0, :, -1]
  # single_traj_train_states = node.train_data[:, :, :-1]
  #
  # print("train controls are", single_traj_train_controls.shape)
  # print("start train states are", single_traj_train_states[0, 0, :])
  # _, train_cost = get_state_trajectory_and_cost(node.hp, node.system,
  #                                               single_traj_train_states[0, 0, :],
  #                                               single_traj_train_controls.squeeze())

  # If there is a final state, also report the defect
  opt_defect = None
  defect = None
  if node.system.x_T is not None:
    opt_defect = node.true_opt_xs[-1] - node.system.x_T
    defect = xs[-1] - node.system.x_T

  plot(hp=node.hp,
       system=node.system,
       data={'x': node.true_opt_xs, 'u': node.true_opt_us, 'other_x': xs, 'other_u': planned_us,
             'cost': node.true_opt_cost, 'defect': opt_defect, 'other_cost': cost, 'other_defect': defect},
       labels={'x': ' (true)',
               'u': ' (true)',
               'other_x': ' (planned)',
               'other_u': ' (planned)'},
       title=title,
       save_as=save_as)


##########################
# Test E2E Node planning #
##########################
def load_node_and_plan(hp, cfg):
  params_path = f'params/{hp.system.name}/e2e_node/'
  plots_path = f'params/{hp.system.name}/e2e_node/'
  Path(params_path).mkdir(parents=True, exist_ok=True)
  Path(plots_path).mkdir(parents=True, exist_ok=True)
  params_names = [f'{i * 50}.p' for i in range(60, 201)]
  plots_names = [f'{i * 50}_epochs.png' for i in range(60, 201)]

  node = NeuralODE(hp, cfg, mle=False)
  true_system = hp.system()  # use the default params here
  true_optimizer = get_optimizer(hp, cfg, true_system)
  node_system = NodeSystem(node=node, true_system=true_system)
  node_optimizer = get_optimizer(hp, cfg, node_system)

  true_solution = true_optimizer.solve()
  true_opt_us = true_solution['u']
  _, true_opt_xs = integrate_time_independent(
    true_system.dynamics, true_system.x_0, true_opt_us, hp.stepsize, hp.num_steps, hp.integration_method)

  for i, params_name in enumerate(params_names):
    try:
      node.load_params(params_path + params_name)
      print("loaded params")
      solution = node_optimizer.solve_with_params(node.params)
      solved_us = solution['u']
      _, integrated_xs = integrate_time_independent(
        true_system.dynamics, true_system.x_0, solved_us, hp.stepsize, hp.num_steps, hp.integration_method)

      plot(hp, true_system,
           data={'x': true_opt_xs,
                 'other_x': integrated_xs,
                 'u': true_opt_us,
                 'other_u': solved_us},
           labels={'x': ' (optimal)',
                   'other_x': ' (learned)',
                   'u': ' (optimal)',
                   'other_u': ' (learned)'},
           styles={'x': '.',
                   'other_x': '-',
                   'u': '.',
                   'other_u': '-'},
           save_as=plots_path + plots_names[i])

    except FileNotFoundError as e:
      print("unable to find the params, so we'll skip")
