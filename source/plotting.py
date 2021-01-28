import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict

from .config import SystemType

plotting_options = {
  SystemType.GLUCOSE: [[0, 1], ["Blood Glucose", "Net Hormonal Concentration"]],
  SystemType.HIVTREATMENT: [[0], ["Healthy cells", "Infected cells", "Viral charge"]]
}


def plot(system,
         data: Dict[str, jnp.ndarray],
         labels: Optional[Dict[str, str]] = None,
         title: Optional[str] = None,
         save_as: Optional[str] = None) -> None:

  sns.set(style='darkgrid')

  if 'adj' not in data:
    height = 7
    num_subplots = 2
  else:
    height = 9
    num_subplots = 3
  plt.figure(figsize=(9, height))
  if title:
    plt.suptitle(title)

  ts_x = jnp.linspace(0, system.T, data['x'].shape[0])
  ts_u = jnp.linspace(0, system.T, data['u'].shape[0])

  plt.subplot(num_subplots, 1, 1)
  if system._type in plotting_options:
    for idx, x_i in enumerate(data['x'].T):
      if idx in plotting_options[system._type][0]:
        plt.plot(ts_x, x_i, label=plotting_options[system._type][1][idx]+labels['x'])
        if 'other_x' in data:
          plt.plot(ts_u, data['other_x'][:, idx], label=plotting_options[system._type][1][idx]+labels['other_x'])
  else:
    plt.plot(ts_x, data['x'], label=labels['x'])
    if 'other_x' in data:
      plt.plot(ts_u, data['other_x'], label=labels['other_x'])
  plt.ylabel("state (x)")
  plt.legend()

  plt.subplot(num_subplots, 1, 2)
  plt.plot(ts_u, data['u'], label=labels['u'])
  if 'other_u' in data:
    plt.plot(ts_u, data['other_u'], label=labels['other_u'])
  plt.ylabel("control (u)")
  plt.legend()

  if 'adj' in data:
    ts_adj = jnp.linspace(0, system.T, data['adj'].shape[0])
    plt.subplot(num_subplots, 1, 3)
    plt.plot(ts_adj, data['adj'], label=labels['adj'])
    plt.ylabel("adjoint (lambda)")
    plt.legend()

  plt.xlabel('time (s)')
  plt.tight_layout()
  if save_as:
    plt.savefig(save_as+".pdf")
  else:
    plt.show()
