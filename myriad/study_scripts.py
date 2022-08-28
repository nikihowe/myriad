# (c) 2021 Nikolaus Howe
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

from jax.config import config
from pathlib import Path

from myriad.defaults import param_guesses
from myriad.neural_ode.create_node import NeuralODE
from myriad.experiments.mle_sysid import run_mle_sysid
from myriad.experiments.node_mle_sysid import run_node_mle_sysid
from myriad.trajectory_optimizers import get_optimizer
from myriad.systems.neural_ode.node_system import NodeSystem
from myriad.systems import get_name
from myriad.useful_scripts import run_trajectory_opt, run_node_trajectory_opt
from myriad.utils import get_state_trajectory_and_cost

config.update("jax_enable_x64", True)


###############
# Noise study #
###############
def study_noise(hp, cfg, experiment_string='mle_sysid'):
  # Parametric, ML
  noise_levels = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1., 2., 5.]

  param_path = f'params/{hp.system.name}/{experiment_string}/'
  plot_path = f'plots/{hp.system.name}/{experiment_string}/'

  hp.num_experiments = 1

  # Run the sysid
  for noise_level in noise_levels:
    hp.noise_level = noise_level
    if experiment_string == 'mle_sysid':
      run_mle_sysid(hp, cfg)
    elif experiment_string == 'node_mle_sysid':
      run_node_mle_sysid(hp, cfg)
    else:
      raise Exception("Didn't recognize experiment string")

  # Make the loss vs noise plot
  costs = []
  defects = []
  # cfg.plot_results = False
  for noise_level in noise_levels:
    param_name = f'noise_{noise_level}_smoothed_{hp.to_smooth}_10_3_3'
    if experiment_string == 'mle_sysid':
      c, d = run_trajectory_opt(hp, cfg, params_path=param_path + param_name + '.p')
    elif experiment_string == 'node_mle_sysid':
      c, d = run_node_trajectory_opt(hp, cfg, params_path=param_path + param_name + '_exp_0.p')
    else:
      raise Exception("Unknown experiment string")
    costs.append(c)
    defects.append(d)

  cd_path = f'costs_and_defects/{hp.system.name}/{experiment_string}/'
  Path(cd_path).mkdir(parents=True, exist_ok=True)
  pkl.dump(noise_levels, open(cd_path + 'noise_levels', 'wb'))
  pkl.dump(costs, open(cd_path + 'costs', 'wb'))
  pkl.dump(defects, open(cd_path + 'defects', 'wb'))

  matplotlib.use("pgf")
  matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
  })
  plt.rcParams["figure.figsize"] = (3.7, 3.1)

  # Get the cost of the truly optimal trajectory
  system = hp.system()
  optimizer = get_optimizer(hp, cfg, system)
  solution = optimizer.solve()
  _, optimal_cost = get_state_trajectory_and_cost(hp, system, system.x_0, solution['u'])

  nl = pkl.load(open(cd_path + 'noise_levels', 'rb'))
  c = pkl.load(open(cd_path + 'costs', 'rb'))
  d = pkl.load(open(cd_path + 'defects', 'rb'))
  plt.plot(nl, c)
  plt.xlabel('noise level')
  plt.ylabel('cost')
  plt.axhline(optimal_cost, color='grey', linestyle='dashed')
  plt.xlim(0, 5)
  plt.grid()
  if d[0] is not None:
    plt.plot(nl, d)
  # plt.title(hp.system.name)
  title = get_name(hp)
  if title is not None:
    plt.suptitle(title)
  plt.savefig(plot_path + 'aanoise_study.pgf', bbox_inches='tight')
  plt.close()

  params_path = f'params/{hp.system.name}/{experiment_string}/guess.p'
  pkl.dump(param_guesses[hp.system], open(params_path, 'wb'))
  c, d = run_trajectory_opt(hp, cfg, params_path=params_path)
  print("c, d", c, d)


def load_system_and_us(hp, cfg, experiment_string, experiment_number):
  system = hp.system()
  optimizer = get_optimizer(hp, cfg, system)
  solution = optimizer.solve()
  us = solution['u']

  if experiment_string is None:
    pass
    # system = hp.system()
    # optimizer = get_optimizer(hp, cfg, system)
    # solution = optimizer.solve()
    # us = solution['u']
    learned_dynamics = system.dynamics

  elif experiment_string == 'mle' or experiment_string == 'e2e':
    params_path = f'params/{hp.system.name}/node_{experiment_string}_sysid/'
    if experiment_string == 'mle':
      params_name = f'noise_{hp.noise_level}_smoothed_{hp.to_smooth}_' \
                    f'{hp.train_size}_{hp.val_size}_{hp.test_size}_exp_{experiment_number}.p'
    else:
      params_name = 'node_e2e.p'

    node = NeuralODE(hp, cfg)
    node.load_params(params_path + params_name)
    print("loaded params", params_path + params_name)
    system = NodeSystem(node, node.system)

    # optimizer = get_optimizer(hp, cfg, system)
    # solution = optimizer.solve_with_params(node.params)
    # us = solution['u']

    def learned_dynamics(x, u, t):
      return system.parametrized_dynamics(node.params, x, u, t)

  else:
    raise Exception("Didn't recognize the experiment string")

  return system, learned_dynamics, us


def study_vector_field(hp, cfg, experiment_string=None, experiment_number=0, title=''):
  matplotlib.use("pgf")
  matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
  })
  plt.rcParams["figure.figsize"] = (4, 3.3)

  num_horizontal_arrows = 21
  num_vertical_arrows = 15

  system, true_dynamics, us = load_system_and_us(hp, cfg, None, 0)
  _, learned_dynamics, _ = load_system_and_us(hp, cfg, experiment_string, 0)
  opt_xs, c = get_state_trajectory_and_cost(hp, system, system.x_0, us)

  # TODO: maybe want to also plot the integrated trajectory

  ts_x = jnp.linspace(0, system.T, opt_xs.shape[0])
  ts_u = jnp.linspace(0, system.T, us.shape[0])

  state_bounds = system.bounds[:hp.state_size].squeeze()

  plt.figure(figsize=(4, 4))

  if experiment_string is None:
    title = 'True Dynamics'
  else:
    title = 'Learned Dynamics'

  plus_title = get_name(hp)
  # if plus_title is not None:
  #   plt.suptitle(plus_title)
  plt.suptitle(title + '  –  ' + plus_title)

  # plt.subplot(2, 1, 1)
  xs, ys = jnp.meshgrid(jnp.linspace(0, system.T, num_horizontal_arrows),
                        jnp.linspace(state_bounds[0], state_bounds[1], num_vertical_arrows))
  xs = xs.flatten()
  ys = ys.flatten()

  times_to_evaluate_at = jnp.linspace(0, system.T, num_horizontal_arrows)
  interpolated_us = jnp.interp(times_to_evaluate_at, ts_u, us.flatten())

  all_true_dynamics = []
  all_learned_dynamics = []

  for i, y in enumerate(ys):
    all_true_dynamics.append(true_dynamics(y, interpolated_us[i % num_horizontal_arrows],
                                           0))  # dynamics are time independent, so can put 0 here
    all_learned_dynamics.append(learned_dynamics(y, interpolated_us[i % num_horizontal_arrows], 0))
  vec_true_y = jnp.array(all_true_dynamics)
  vec_learned_y = jnp.array(all_learned_dynamics)
  vec_x = jnp.ones_like(vec_true_y)

  plt.quiver(xs, ys, vec_x, vec_true_y, angles='xy', width=0.003, alpha=0.9, color='blue', label='True Dynamics')
  plt.quiver(xs, ys, vec_x, vec_learned_y, angles='xy', width=0.003, alpha=0.9, color='orange',
             label='Learned Dynamics')

  # Also plot the true dynamics
  plt.plot(ts_x, opt_xs, label='True Trajectory', lw=1, ls='--', c='grey')
  plt.grid()
  plt.ylim(state_bounds)
  plt.xlim((0., system.T))
  # plt.plot(ts_x, opt_xs, label='State')
  # arrow = plt.arrow(0, 0, 0.5, 0.6)
  # plt.legend([arrow, ], ['My label', ])
  plt.legend(loc='upper right', fontsize=8, title_fontsize=10)
  plt.ylabel('state (x)')

  # plt.subplot(2, 1, 2)
  #
  # plt.plot(ts_u, us, label='Control')
  # plt.grid()
  # plt.xlabel('time (s)')
  # plt.ylabel('control (u)')
  # # plt.ylim((0., max(us)))
  # plt.xlim((0., system.T))

  if experiment_string is None:
    plot_path = f'plots/{hp.system.name}/true/'
  else:
    plot_path = f'plots/{hp.system.name}/node_{experiment_string}_sysid/'
  Path(plot_path).mkdir(parents=True, exist_ok=True)

  plt.tight_layout()
  plt.savefig(plot_path + f'{hp.system.name}_{experiment_string}_vector_study.{cfg.file_extension}',
              bbox_inches='tight')
  plt.close()
