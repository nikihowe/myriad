# (c) 2021 Nikolaus Howe
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pickle as pkl

from dataclasses import dataclass
from jax import config
from typing import Optional

from myriad.config import HParams, Config, SamplingApproach
from myriad.trajectory_optimizers import get_optimizer
from myriad.utils import get_state_trajectory_and_cost, generate_dataset, yield_minibatches

config.update("jax_enable_x64", True)


def make_empty_losses():
  return {'ts': [],
          'train_loss': [],
          'validation_loss': [],
          'loss_on_opt': [],
          'control_costs': [],
          'constraint_violation': [],
          'divergence_from_optimal_us': [],
          'divergence_from_optimal_xs': []}


##############################
# Neural ODE for opt control #
##############################
@dataclass
class NeuralODE(object):
  hp: HParams
  cfg: Config
  key: jnp.ndarray = jax.random.PRNGKey(42)
  mle: bool = True
  dataset: Optional[jnp.ndarray] = None

  def __post_init__(self) -> None:
    self.system = self.hp.system()

    self.num_steps = self.hp.intervals * self.hp.controls_per_interval
    self.stepsize = self.system.T / self.num_steps  # Segment length

    # Get the true optimal controls and corresponding trajectory
    # real_solver = self.hp.nlpsolver
    # self.hp.nlpsolver = NLPSolverType.SLSQP
    if self.mle:
      # Try to load the optimal trajectories. If they don't exist, solve for them ourselves.
      opt_path = f'datasets/{self.hp.system.name}/optimal_trajectories/'
      Path(opt_path).mkdir(parents=True, exist_ok=True)
      opt_name = f'{self.hp.intervals}_{self.hp.controls_per_interval}_{self.hp.optimizer.name}_' \
                 f'{self.hp.integration_method.name}_{self.hp.quadrature_rule.name}'
      try:
        self.true_opt_us = jnp.array(pkl.load(open(f'{opt_path + opt_name}_us', 'rb')))
        self.true_opt_xs = jnp.array(pkl.load(open(f'{opt_path + opt_name}_xs', 'rb')))
      except FileNotFoundError as e:
        print("Didn't find pre-saved optimal trajectories, so calculating our own.")
        self.optimizer = get_optimizer(self.hp, self.cfg, self.system)
        self.optimal_solution = self.optimizer.solve()
        self.true_opt_us = self.optimal_solution['u']
        self.true_opt_xs = self.optimal_solution['x']
        pkl.dump(self.true_opt_us, open(f'{opt_path + opt_name}_us', 'wb'))
        pkl.dump(self.true_opt_xs, open(f'{opt_path + opt_name}_xs', 'wb'))
      # TODO: think about quadratic case
      # _, self.true_opt_xs = self.integrate(self.true_opt_us)
      # print("getting state traj and cost")
      self.true_opt_xs, self.true_opt_cost = get_state_trajectory_and_cost(
        self.hp, self.system, self.system.x_0, self.true_opt_us)

      self.true_x_and_u_opt = jnp.concatenate([self.true_opt_xs, self.true_opt_us], axis=1)
    # self.hp.nlpsolver = real_solver

    # Create a best guess which we'll update as we plan
    self.best_guess_us = None
    self.best_guess_us_cost = None

    # Record the important info about this node
    self.info = f"{self.hp.learning_rate}" \
                f"_{self.hp.train_size}" \
                f"_{self.hp.val_size}" \
                f"_{self.hp.test_size}" \
                f"_start_spread_{self.hp.start_spread}" \
                f"_{self.hp.minibatch_size}" \
                f"_({'_'.join(str(layer) for layer in self.hp.hidden_layers)})" \
                f"_{self.hp.sample_spread}" \
                f"_{self.hp.noise_level}"

    # Generate the (initial) dataset
    self.train_data, self.validation_data, self.test_data, self.full_data = self.make_datasets(first_time=True)

    # Initialize the parameters and optimizer state
    self.net = hk.without_apply_rng(hk.transform(self.net_fn))
    mb = next(yield_minibatches(self.hp, self.hp.train_size, self.train_data))
    print("node: params initialized with: ", mb[1, 1, :].shape)
    self.key, subkey = jax.random.split(self.key)  # Always update the NODE's key
    self.params = self.net.init(subkey, mb[1, 1, :])
    self.opt = optax.adam(self.hp.learning_rate)
    self.opt_state = self.opt.init(self.params)
    self.losses = make_empty_losses()
    if self.cfg.verbose:
      print("node: minibatches are of shape", mb.shape)
      print("node: initialized network weights")

  # The neural net for the neural ode: a small and simple MLP
  def net_fn(self, x_and_u: jnp.array) -> jnp.array:
    the_layers = []
    for layer_size in self.hp.hidden_layers:
      the_layers.append(hk.Linear(layer_size))
      the_layers.append(jax.nn.sigmoid)
    the_layers.append(hk.Linear(len(self.system.x_0)))
    mlp = hk.Sequential(the_layers)
    return mlp(x_and_u)  # will automatically broadcast over minibatches

  def save_params(self, filename: str) -> None:
    pkl.dump(self.params, open(filename, 'wb'))

  def load_params(self, params_pickle: str) -> None:
    try:
      temp_params = hk.data_structures.to_mutable_dict(pkl.load(open(params_pickle, 'rb')))
      print("loaded node params from file")
      if 'linear/~/linear' in temp_params:
        temp_params['linear_1'] = temp_params['linear/~/linear']
        del temp_params['linear/~/linear']
      if 'linear/~/linear/~/linear' in temp_params:
        temp_params['linear_2'] = temp_params['linear/~/linear/~/linear']
        del temp_params['linear/~/linear/~/linear']
      self.params = hk.data_structures.to_immutable_dict(temp_params)
    except FileNotFoundError as e:
      raise e

  def load_dataset(self, file_path: str) -> None:
    try:
      dataset = pkl.load(open(file_path, 'rb'))
      dataset = jnp.array(dataset)
      self.train_data = dataset[:self.hp.train_size]
      self.validation_data = dataset[self.hp.train_size:self.hp.train_size + self.hp.val_size]
      self.test_data = dataset[self.hp.train_size + self.hp.val_size:]
      self.all_data = dataset
    except FileNotFoundError as e:
      raise e

  def make_datasets(self, first_time=False):
    # Generate the new data
    if self.hp.sampling_approach == SamplingApproach.CURRENT_OPTIMAL and self.best_guess_us is not None:
      all_data = generate_dataset(self.hp, self.cfg, given_us=self.best_guess_us)
    else:
      all_data = generate_dataset(self.hp, self.cfg)

    # Split the new data
    train_data = all_data[:self.hp.train_size]
    validation_data = all_data[self.hp.train_size:self.hp.train_size + self.hp.val_size]
    test_data = all_data[self.hp.train_size + self.hp.val_size:]

    # If not first time, add the new data to our existing dataset
    if not first_time:
      train_data = jnp.concatenate((self.train_data, train_data), axis=0)
      validation_data = jnp.concatenate((self.validation_data, validation_data), axis=0)
      test_data = jnp.concatenate((self.test_data, test_data), axis=0)

    if self.cfg.verbose:
      print("Generated training trajectories of shape", train_data.shape)
      print("Generated validation trajectories of shape", validation_data.shape)
      print("Generated test trajectories of shape", test_data.shape)

    return train_data, validation_data, test_data, all_data

  def augment_datasets(self):
    self.train_data, self.validation_data, self.test_data, self.full_dataset = self.make_datasets(first_time=False)


if __name__ == "__main__":
  hp = HParams()
  cfg = Config()
  my_node = NeuralODE(hp, cfg)
  print("my_node", my_node)
