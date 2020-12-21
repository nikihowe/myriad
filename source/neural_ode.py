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

import random as rd
from absl import app, flags
import pickle

from typing import Any, Generator, Mapping, Tuple, Optional
Batch = Mapping[str, jnp.array]
OptState = Any

from source.config import HParams, Config, SystemType, OptimizerType, IntegrationOrder
from source.optimizers import TrajectoryOptimizer, get_optimizer
from source.systems import FiniteHorizonControlSystem, get_system
from source.utils import integrate, integrate_in_parallel

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


def make_neural_ode(
  hp: HParams,
  cfg: Config,
  learning_rate: jnp.float32 = 0.001,
  num_training_steps: jnp.float32 = 10_001,
  use_params: Optional[str] = None,
  train_size: jnp.int32 = 100_000,
  mb_size: jnp.int32 = 128,
  order: IntegrationOrder = IntegrationOrder.CONSTANT, # only constant and linear work for now
  learning_approach: str = "simple"
  ):

  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)
  stepsize = system.T / hp.intervals  # Segment length

  # NOTE: for now, this will only work with collocation, and scalar controls
  # though of course the integration is rk4 family
  def get_many_xu_trajectories(key=random.PRNGKey(42)):
    key, subkey = random.split(key)

    # We need upper and lower bounds in order to generate random control
    # vectors for training. 
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    print("u lower", u_lower)
    print("u upper", u_upper)
    if system._type == SystemType.CANCER:
      u_upper = 2.
    train_us = random.uniform(subkey, (train_size, hp.intervals + 1), minval=u_lower, maxval=u_upper)
    if system._type == SystemType.VANDERPOL: # avoid vanderpol explosion
      train_us = train_us * 0.1
    # Integrate all the trajectories in parallel, starting from the start state,
    # and applying the randomly chosen controls, different for each trajectory
    _, train_xxs = integrate_in_parallel(system.dynamics,
                                         system.x_0[jnp.newaxis].repeat(train_size, axis=0),
                                         train_us, stepsize, hp.intervals, None, order)
    xs_and_us = jnp.concatenate([train_xxs, train_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    # print("returning shape", xs_and_us.shape)
    # print("training trajectories", xs_and_us[:10])
    return xs_and_us

  train_trajectories = get_many_xu_trajectories()
  print("generated training trajectories!")

  def get_minibatch(i=0):
    i = i % int(len(train_trajectories) / (mb_size + 1))
    return train_trajectories[mb_size*i:mb_size*(i+1)]

  def get_informed_minibatch(controls, key=random.PRNGKey(43)):
    noise = random.normal(key=key, shape=controls.shape)
    _, train_xxs = integrate_in_parallel(system.dynamics,
                                         system.x_0[jnp.newaxis].repeat(train_size, axis=0),
                                         train_us, stepsize, hp.intervals, None, order)
    xs_and_us = jnp.concatenate([train_xxs, train_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)


  def imitation_loss(true_xs, found_xs):
    return jnp.linalg.norm(true_xs, found_xs)

  def net_fn(x_and_u: jnp.array) -> jnp.array: # going to be a 1D array for now
    # print("x and u shape", x_and_u.shape)
    mlp = hk.Sequential([
        hk.Linear(40), nn.relu,
        hk.Linear(40), nn.relu,
        hk.Linear(len(system.x_0)),
    ])
    out = mlp(x_and_u)
    # print("out shape", out.shape)
    return mlp(x_and_u) # will be automatically broadcast

  # Initialize the network
  net = hk.without_apply_rng(hk.transform(net_fn))
  mb = get_minibatch()
  print("minibatch", mb.shape)
  # print("state shape", mb['xs'].shape)
  # print("controls shape", mb['us'].shape)
  params = net.init(random.PRNGKey(42), mb[0, 0, :])
  opt = optax.adam(learning_rate)
  opt_state = opt.init(params)
  print("initialized")

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
    def apply_net(x, u):
      return net.apply(params, jnp.append(x, u))

    # print("x0 shape", system.x_0[jnp.newaxis].repeat(mb_size, axis=0).shape)
    # print("batch shape", batch.shape)
    # Only take the controls from the batch
    controls = batch[:, :, -1]
    # print("controls shape", controls.shape)
    _, predicted_states = integrate_in_parallel(apply_net,
                                             system.x_0[jnp.newaxis].repeat(mb_size, axis=0),
                                             controls, stepsize, hp.intervals, None, order)
    # print("got predcted", predicted_states.shape)
    # print("predicted values", predicted_states[0])

    true_states = batch[:, :, :-1]
    # print("true states", true_states.shape)
    # print("true states values", true_states[0])

    # Deal with simulator explosion
    # TODO: there must be a better way, no?
    true_states = jnp.nan_to_num(true_states)
    predicted_states = jnp.nan_to_num(predicted_states)

    loss = jnp.nan_to_num(jnp.mean(jnp.nan_to_num((predicted_states - true_states)*(predicted_states - true_states))))
    return loss

  def load_params(my_pickle):
    p = None
    try:
      p = pickle.load(open(my_pickle, "rb"))
    except:
      print("Unable to find file '{}'".format(my_pickle))
      raise SystemExit
    return p
 
  # If use_params, then load the params from the file.
  # Otherwise, do training according to the chosen strategy.
  #
  # Strategies currently include:
  # - uniform random
  #     random exploration
  # - adaptive random (not yet implemented)
  #     every X steps, choose a new set of sample controls
  #     by sampling around the current best-performing controls
  # 
  if use_params:
    params = load_params(use_params)
    print("loaded params")
    print(params.keys())
  else:
    if learning_approach == "simple":
      old_loss = -1
      for step in trange(num_training_steps):
        if step % 1000 == 0:
          cur_loss = loss(params, get_minibatch(step))
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss
        params, opt_state = update(params, opt_state, get_minibatch(step))
    else: # use informed learning approach
      # NOTE: this is not yet implemented
      raise NotImplementedError
      # old_loss = -1
      # for step in trange(num_training_steps):
      #   if step % 10 == 0:
      #     system.dynamics = jit(apply_net)
      #     new_optimizer = get_optimizer(hp, cfg, system)
      #     if new_optimizer.require_adj:
      #       x, u, adj = new_optimizer.solve()
      #     else:
      #       x, u = new_optimizer.solve()

      #     cur_loss = loss(params, get_minibatch(step))
      #     print(step, cur_loss)
      #     if jnp.allclose(cur_loss, old_loss):
      #       break
      #     old_loss = cur_loss
      #   params, opt_state = update(params, opt_state, get_minibatch(step))


  def apply_net(x, u):
    return net.apply(params, jnp.append(x, u))
  
  # Given a set of controls and corresponding true states, plots them,
  # and also plots the states as calculated by the NODE using those controls.
  # If get_optimal_trajectory, uses the optimal controls and corresponding
  # states, instead of those passed as arguments
  def plot_trajectory(get_optimal_trajectory: bool = False,
                      test_state: jnp.ndarray = train_trajectories[-1, :, :-1],
                      test_controls: jnp.ndarray = train_trajectories[-1, :, -1]):

    if get_optimal_trajectory:
      # Also get the optimal behaviour
      if optimizer.require_adj:
          x, u, adj = optimizer.solve()
          _, predicted_states = integrate(apply_net, system.x_0, u,
                                          stepsize, hp.intervals, None, order)

          system.plot_solution(x, u, adj, other_x=predicted_states, other_u=None)
      else:
          x, u = optimizer.solve()
          _, predicted_states = integrate(apply_net, system.x_0, u,
                                          stepsize, hp.intervals, None, order)

          system.plot_solution(x, u, other_x=predicted_states, other_u=None)
      _, predicted_states = integrate(apply_net, system.x_0, test_controls, stepsize, hp.intervals, None, order)

    else:
      # print("the predicted states are", predicted_states[:10])
      system.plot_solution(test_state, test_controls, other_x=predicted_states, other_u=None)

  # First, get the optimal controls and resulting trajectory using the true system model.
  # Then, replace the model dynamics with the trained neural network,
  # and use that to find "optimal" controls.
  # Finally, get the resulting true state trajectory coming from those suboptimal controls.
  def plan_with_model():
    # Now do planning with the learned model
    if optimizer.require_adj:
      true_x, true_u, true_adj = optimizer.solve()
    else:
      true_x, true_u = optimizer.solve()

    # Replace the system dynamics with the learned dynamics
    # Note that x is the "dreamt" trajectory, calculated using the NODE dynamics.
    system.dynamics = jit(apply_net)
    new_optimizer = get_optimizer(hp, cfg, system)
    if new_optimizer.require_adj:
      x, u, adj = new_optimizer.solve()
    else:
      x, u = new_optimizer.solve()

    # Now get the true trajectory followed when we apply these controls
    _, x = integrate(apply_net, system.x_0, u,
                     stepsize, hp.intervals, None, order)

    # Plot
    if new_optimizer.require_adj:
      system.plot_solution(true_x, true_u, true_adj, other_x=x, other_u=u)
    else:
      system.plot_solution(true_x, true_u, other_x=x, other_u=u)

  plot_trajectory(get_optimal_trajectory=True)
  # plan_with_model()

  return params

def run_net(hp: HParams, cfg: Config, use_params: Optional[str] = None, save: bool = False):
  # train_data = get_train_data()
  # make_normal_neural_net(train_data)
  ############
  # parser = simple_parsing.ArgumentParser()
  # parser.add_arguments(HParams, dest="hparams")
  # parser.add_arguments(Config, dest="config")

  # args = parser.parse_args()
  # hp = args.hparams
  # cfg = args.config

  # # Set our seeds for reproducibility
  # rd.seed(hp.seed)
  # np.random.seed(hp.seed)

  # # Load config, then build system
  # gin_files = ['./source/gin-configs/default.gin']
  # gin_bindings = FLAGS.gin_bindings
  # gin.parse_config_files_and_bindings(gin_files,
  #                                     bindings=gin_bindings,
  #                                     skip_unknown=False)
  # print(hp)
  # print(cfg)
  print("running net!")
  params = make_neural_ode(hp, cfg, num_training_steps=10_000, use_params=use_params)

  if save:
    from datetime import date
    date_string = date.today().strftime("%Y-%m-%d")
    pickle.dump(params, open("source/params/{}_{}.p".format(hp.system.name, date_string), "wb"))

  # favorite_color = pickle.load(open("save.p", "rb"))