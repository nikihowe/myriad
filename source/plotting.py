import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Dict

from .config import SystemType

state_descriptions = {
  SystemType.CARTPOLE: [[0, 1, 2, 3], ["Position", "Angle", "Velocity", "Angular velocity"]],
  SystemType.SEIR: [[0, 1, 2, 3], ["S", "E", "I", "N"]],
  SystemType.TUMOUR: [[0, 1, 2], ["p", "q", "y"]],
  SystemType.VANDERPOL: [[0, 1], ["x0", "x1"]],
  SystemType.BACTERIA: [[0], ["Bacteria concentration"]],
  SystemType.BEARPOPULATIONS: [[0, 1, 2], ["Park population", "Forest population", "Urban population"]],
  SystemType.BIOREACTOR: [[0], ["Bacteria concentration"]],
  SystemType.CANCER: [[0], ["Normalized tumour density"]],
  SystemType.EPIDEMICSEIRN: [[2], ["Susceptible population", "Exposed population",
                                   "Infectious population", "Total population"]],
  SystemType.FISHHARVEST: [[0], ["Fish population"]],
  SystemType.GLUCOSE: [[0, 1], ["Blood glucose", "Net hormonal concentration"]],
  SystemType.HIVTREATMENT: [[0], ["Healthy cells", "Infected cells", "Viral charge"]],
  SystemType.INVASIVEPLANT: [[0, 1, 2, 3, 4], ["Focus 1", "Focus 2", "Focus 3", "Focus 4", "Focus 5"]],
  SystemType.MOULDFUNGICIDE: [[0], ["Mould population"]],
  SystemType.PREDATORPREY: [[0, 1], ["Predator population", "Prey population"]],
  # SystemType.SIMPLECASE
  # SystemType.SIMPLECASEWITHBOUNDS
  SystemType.TIMBERHARVEST: [[0], ["Cumulative timber harvested"]]
}

# NOTE: the control descriptions are currently not used for plotting
control_descriptions = {
  SystemType.CARTPOLE: [[0], ["Force applied to cart"]],
  SystemType.SEIR: [[0], ["Response intensity"]],
  SystemType.TUMOUR: [[0], ["Drug strength"]],
  SystemType.VANDERPOL: [[0], ["Control"]],
  SystemType.BACTERIA: [[0], ["Amount of chemical nutrient"]],
  SystemType.BEARPOPULATIONS: [[0, 1], ["Harvesting rate in park", "Harvesting rate in forest"]],
  SystemType.BIOREACTOR: [[0], ["Amount of chemical nutrient"]],
  SystemType.CANCER: [[0], ["Drug strength"]],
  SystemType.EPIDEMICSEIRN: [[0], ["Vaccination rate"]],
  SystemType.FISHHARVEST: [[0], ["Harvest rate"]],
  SystemType.GLUCOSE: [[0], ["Insulin level"]],
  SystemType.HIVTREATMENT: [[0], ["Drug intensity"]],
  SystemType.MOULDFUNGICIDE: [[0], ["Fungicide level"]],
  SystemType.PREDATORPREY: [[0], ["Pesticide level"]],
  # SystemType.SIMPLECASE
  # SystemType.SIMPLECASEWITHBOUNDS
  SystemType.TIMBERHARVEST: [[0], ["Reinvestment level"]]
}


def plot(system,
         data: Dict[str, jnp.ndarray],
         labels: Optional[Dict[str, str]] = None,
         title: Optional[str] = None,
         save_as: Optional[str] = None) -> None:

  sns.set(style='darkgrid')

  # Separate plotting for the discrete-time system
  if system._type == SystemType.INVASIVEPLANT:
    system.plot_solution(data['x'], data['u'], data['adj'])
    return
  elif system._type == SystemType.SEIR:
    system.plot_solution(data['x'], data['u'])
    return

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

  # Every system except SIMPLECASE and SIMPLECASEWITHBOUNDS
  # Plot exactly those state columns which we want plotted
  plt.subplot(num_subplots, 1, 1)
  if system._type in state_descriptions:
    for idx, x_i in enumerate(data['x'].T):
      if idx in state_descriptions[system._type][0]:
        plt.plot(ts_x, x_i, label=state_descriptions[system._type][1][idx])
        if 'other_x' in data:
          plt.plot(ts_u, data['other_x'][:, idx], label=state_descriptions[system._type][1][idx])
  else:
    plt.plot(ts_x, data['x'], label=labels['x'])
    if 'other_x' in data:
      plt.plot(ts_u, data['other_x'], label=labels['other_x'])
  plt.ylabel("state (x)")
  plt.legend()

  # Same thing as above, but for the controls
  plt.subplot(num_subplots, 1, 2)
  if system._type in control_descriptions:
    for idx, u_i in enumerate(data['u'].T):
      if idx in control_descriptions[system._type][0]:
        plt.plot(ts_u, u_i, label=control_descriptions[system._type][1][idx])
        if 'other_u' in data:
          plt.plot(ts_u, data['other_u'][:, idx], label=control_descriptions[system._type][1][idx])
  else:
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
