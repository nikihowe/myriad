import jax.numpy as jnp
from jax import jit, grad, tree_multimap, vmap, random, nn
from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import optax

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time
import simple_parsing

from typing import Any, Generator, Mapping, Tuple
Batch = Mapping[str, jnp.array]
OptState = Any

from source.config import HParams, Config, SystemType, OptimizerType
from source.optimizers import get_optimizer
from source.systems import get_system

def get_train_data(train_size=10000, key=random.PRNGKey(41)):
  key, subkey = random.split(key)
  key, subkey1 = random.split(key)
  train_x = random.uniform(subkey, (train_size,)) * 2 * jnp.pi
  train_y = jnp.sin(train_x) + random.uniform(subkey1, (train_size,)) / 2
  return train_x, train_y

def make_normal_neural_net(train_data, learning_rate=0.001, layers=None):

  def get_minibatch(i=0):
    train_x, train_y = train_data
    i = i % int(len(train_data) / 128 + 1)
    chosen_xs = train_x[128*i:128*(i+1)].reshape(-1, 1)
    chosen_ys = train_y[128*i:128*(i+1)].reshape(-1, 1)
    return {'x': chosen_xs, 'y': chosen_ys}

  def net_fn(x: jnp.array) -> jnp.array: # going to be a 1D array for now
    mlp = hk.Sequential([
        hk.Linear(40), nn.relu,
        hk.Linear(40), nn.relu,
        hk.Linear(1),
    ])
    return mlp(x) # will be automatically broadcast

  net = hk.without_apply_rng(hk.transform(net_fn))
  mb = get_minibatch()
  params = net.init(random.PRNGKey(42), mb['x'])
  opt = optax.adam(learning_rate)
  opt_state = opt.init(params)
  print("initialized")

  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    x = batch['x'].astype(jnp.float32)
    y = batch['y'].astype(jnp.float32)
    logits = net.apply(params, x)
    # print("y", y.shape)
    # print("logits", logits.shape)
    loss = 1/len(y) * ((y - logits).T @ (y - logits)).squeeze()
    return loss

  @jit
  def update(
      params: hk.Params,
      opt_state: OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    # print("params", params)
    """Learning rule (adam)."""
    grads = grad(loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  old_loss = 1000
  for step in trange(10_001):
    if step % 1000 == 0:
      cur_loss = loss(params, get_minibatch(step))
      print(step, cur_loss)
      if jnp.allclose(cur_loss, old_loss):
        break
      old_loss = cur_loss
    params, opt_state = update(params, opt_state, get_minibatch(step))

  def make_plot(test_size=100):
    test_x = np.random.rand(test_size) * 2 * np.pi
    test_y = np.sin(test_x) + np.random.rand(test_size)/2
    batch = {'x': test_x.reshape(test_size, 1),
            'y': test_y.reshape(test_size, 1)}
    pred_y = net.apply(params, batch['x'])
    plt.plot(test_x, test_y, 'o')
    plt.plot(test_x, pred_y, 'o')
    plt.show()

  make_plot()

def make_neural_ode():
  def net_fn(x: jnp.array) -> jnp.array: # going to be a 1D array for now
    mlp = hk.Sequential([
        hk.Linear(40), nn.relu,
        hk.Linear(40), nn.relu,
        hk.Linear(1),
    ])
    return mlp(x) # will be automatically broadcast

  @jit
  def update(
      params: hk.Params,
      opt_state: OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    # print("params", params)
    """Learning rule (adam)."""
    grads = grad(loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:

    logits = net.apply(params, x)
    # print("y", y.shape)
    # print("logits", logits.shape)
    loss = 1/len(y) * ((y - logits).T @ (y - logits)).squeeze()
    return loss

  pass

def run_net():
  train_data = get_train_data()
  make_normal_neural_net(train_data)
  ############
  # parser = simple_parsing.ArgumentParser()
  # parser.add_arguments(HParams, dest="hparams")
  # parser.add_arguments(Config, dest="config")
  # args = parser.parse_args()
  # hp = args.hparams
  # cfg = args.config
  # print(hp)
  # print(cfg)
  # print(hp)