# (c) 2021 Nikolaus Howe
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import AnchoredText
from typing import Dict, Optional, Tuple

from myriad.config import SystemType, IntegrationMethod, OptimizerType, HParams
from myriad.systems import state_descriptions, control_descriptions
from myriad.systems import get_name


def plot_losses(hp, path_to_csv, save_as=None):
  etv = np.genfromtxt(path_to_csv, delimiter=',')
  if len(etv) == 10000:  # TODO: remove except for ne2e
    print("clipping to 9999")
    etv = etv[:-1]
  epochs = etv[:, 0]
  train = etv[:, 1]
  val = etv[:, 2]
  if save_as is not None and save_as.endswith(('pgf', 'pdf')):
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
      "pgf.texsystem": "pdflatex",
      'font.family': 'serif',
      'text.usetex': True,
      'pgf.rcfonts': False,
    })
  title = get_name(hp)
  print("title is", title)
  plt.figure(figsize=(4.5, 3.5))

  fixed_epochs = []
  transitions = []
  offset = 0
  previous = 0
  for i, epoch in enumerate(epochs):
    if i > 0 and epoch == 0:
      offset += previous
      transitions.append(epoch + offset)
    fixed_epochs.append(epoch + offset)
    previous = epoch

  plt.plot(fixed_epochs, train, label='train loss')
  plt.plot(fixed_epochs, val, label='validation loss')
  if title is not None:
    plt.title(title)

  for transition in transitions:
    plt.axvline(transition, linestyle='dashed', color='grey')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.grid()
  plt.legend()
  plt.tight_layout()
  plt.yscale('log')
  if save_as is not None:
    plt.savefig(save_as, bbox_inches='tight')
    plt.close()
  else:
    plt.show()


def plot_result(result, hp, save_as=None):
  adj = len(result) == 3
  data = {}
  if adj:
    x_guess, u_guess, adj_guess = result
    data['adj'] = adj_guess
  else:
    x_guess, u_guess = result
  data['x'] = x_guess
  data['u'] = u_guess
  plot(hp, hp.system(), data, save_as=save_as)


def plot(hp, system,
         data: Dict[str, jnp.ndarray],
         labels: Optional[Dict[str, str]] = None,
         styles: Optional[Dict[str, str]] = None,
         widths: Optional[Dict[str, float]] = None,
         title: Optional[str] = None,
         save_as: Optional[str] = None,
         figsize: Optional[Tuple[float, float]] = None) -> None:
  if save_as is not None and save_as.endswith(('pgf', 'pdf')):  # comment out for the cluster
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
      "pgf.texsystem": "pdflatex",
      'font.family': 'serif',
      'text.usetex': True,
      'pgf.rcfonts': False,
    })

  if styles is None:
    styles = {}
    for name in data:
      styles[name] = '-'

  if widths is None:
    widths = {}
    for name in data:
      widths[name] = 1.

  # Separate plotting for the discrete-time system
  if hp.system == SystemType.INVASIVEPLANT:
    system.plot_solution(data['x'], data['u'], data['adj'])
    return
  # elif hp.system == SystemType.SEIR:
  #   system.plot_solution(data['x'], data['u'])
  #   return

  if figsize is not None:
    plt.figure(figsize=figsize)
  else:
    # plt.rcParams["figure.figsize"] = (4, 3.3)
    plt.figure(figsize=(4, 4))

  # if 'adj' not in data:
  #   height = 4#5.6
  #   num_subplots = 2
  # else:
  #   height = 9
  #   num_subplots = 3
  # # plt.figure(figsize=(7, height))
  # plt.figure(figsize=(5, height))
  num_subplots = 2
  title = get_name(hp)
  if title is not None:
    plt.suptitle(title)
  # else:
  #   if hp.optimizer == OptimizerType.COLLOCATION:
  #     plt.suptitle(
  #       f'{hp.system.name}') # {hp.optimizer.name}:{hp.intervals} {hp.quadrature_rule.name} {hp.integration_method.name}')
  #   else:
  #     plt.suptitle(
  #       f'{hp.system.name}') # {hp.optimizer.name}:{hp.intervals}x{hp.controls_per_interval} {hp.integration_method.name}')

  order_multiplier = 2 if hp.integration_method == IntegrationMethod.RK4 else 1

  ts_x = jnp.linspace(0, system.T, data['x'].shape[0])
  ts_u = jnp.linspace(0, system.T, data['u'].shape[0])

  # Every system except SIMPLECASE and SIMPLECASEWITHBOUNDS
  # Plot exactly those state columns which we want plotted
  plt.subplot(num_subplots, 1, 1)
  if hp.system in state_descriptions:
    for idx, x_i in enumerate(data['x'].T):
      if idx in state_descriptions[hp.system][0]:
        plt.plot(ts_x, x_i, styles['x'], lw=widths['x'],
                 label=state_descriptions[hp.system][1][idx] + labels['x'])
        if 'other_x' in data:
          plt.plot(jnp.linspace(0, system.T, data['other_x'][:, idx].shape[0]),
                   data['other_x'][:, idx], styles['other_x'], lw=widths['other_x'],
                   label=state_descriptions[hp.system][1][idx] + labels['other_x'])
  else:
    plt.plot(ts_x, data['x'], styles['x'], lw=widths['x'], label=labels['x'])
    if 'other_x' in data:
      plt.plot(jnp.linspace(0, system.T, data['other_x'].shape[0]),
               data['other_x'], styles['other_x'], lw=widths['other_x'], label=labels['other_x'])
  plt.ylabel("state (x)")
  plt.grid()
  plt.legend(loc="upper left")

  # Same thing as above, but for the controls
  ax = plt.subplot(num_subplots, 1, 2)
  if hp.system in control_descriptions:
    for idx, u_i in enumerate(data['u'].T):
      if idx in control_descriptions[hp.system][0]:
        plt.plot(ts_u, u_i, styles['u'], lw=widths['u'], label=control_descriptions[hp.system][1][idx] + labels['u'])
        if 'other_u' in data and data['other_u'] is not None:
          plt.plot(jnp.linspace(0, system.T, data['other_u'][:, idx].shape[0]),
                   data['other_u'][:, idx], styles['other_u'], lw=widths['other_u'],
                   label=control_descriptions[hp.system][1][idx] + labels['other_u'])
  else:
    plt.plot(ts_u, data['u'], styles['u'], lw=widths['u'], label=labels['u'])
    if 'other_u' in data:
      plt.plot(jnp.linspace(0, system.T, data['other_u'].shape[0]),
               data['other_u'], styles['other_u'], lw=widths['other_u'], label=labels['other_u'])
  plt.ylabel("control (u)")
  plt.grid()
  plt.legend(loc="upper left")

  if 'cost' in data and 'other_cost' not in data:
    cost_text = f"Cost: {data['cost']:.2f}"
    if 'defect' in data and data['defect'] is not None:
      for i, d in enumerate(data['defect']):
        if i == 0:
          cost_text += f"\nDefect: {d:.2f}"
        else:
          cost_text += f" {d:.2f}"

    at = AnchoredText(cost_text,
                      prop=dict(size=10), frameon=False,
                      loc='upper right',
                      )
    # at.set_alpha(0.5)
    # at.patch.set_alpha(0.5)

    at.txt._text.set_bbox(dict(facecolor="#FFFFFF", edgecolor="#DBDBDB", alpha=0.7))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at, )

  elif 'cost' in data and 'other_cost' in data:
    cost_text = f"Optimal cost: {data['cost']:.2f}"
    if 'defect' in data and data['defect'] is not None:
      for i, d in enumerate(data['defect']):
        if i == 0:
          cost_text += f"\nOptimal defect: {d:.2f}"
        else:
          cost_text += f" {d:.2f}"

    cost_text += f"\nAchieved cost: {data['other_cost']:.2f}"
    if 'other_defect' in data and data['other_defect'] is not None:
      for i, d in enumerate(data['other_defect']):
        if i == 0:
          cost_text += f"\nAchieved defect: {d:.2f}"
        else:
          cost_text += f"  {d:.2f}"

    at = AnchoredText(cost_text,
                      prop=dict(size=10), frameon=False,
                      loc='upper right',
                      )
    # at.set_alpha(0.5)
    # at.patch.set_alpha(0.5)

    at.txt._text.set_bbox(dict(facecolor="#FFFFFF", edgecolor="#DBDBDB", alpha=0.7))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at, )

  if 'adj' in data:
    ts_adj = jnp.linspace(0, system.T, data['adj'].shape[0])
    plt.subplot(num_subplots, 1, 3)
    if labels is not None and 'adj' in labels:
      plt.plot(ts_adj, data['adj'], label=labels['adj'])
    else:
      plt.plot(ts_adj, data['adj'], label='Adjoint')
    plt.ylabel("adjoint (lambda)")
    plt.legend(loc="upper left")

  plt.xlabel('time (s)')
  plt.tight_layout()
  if save_as:
    plt.savefig(save_as, bbox_inches='tight')
    plt.close()
  else:
    plt.show()


if __name__ == "__main__":
  hp = HParams()
  path_to_csv = f'../losses/{hp.system.name}/1_1_1'
  plot_losses(path_to_csv, save_as=f'../plots/{hp.system.name}/1_1_1/{hp.system.name}_train.pdf')
  plot_losses(path_to_csv, save_as=f'../plots/{hp.system.name}/1_1_1/{hp.system.name}_train.pgf')
  # plot_losses(path_to_csv)
