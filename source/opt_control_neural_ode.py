import jax.numpy as jnp
from jax import jit, grad, tree_multimap, vmap, random, nn
from jax.flatten_util import ravel_pytree
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
from datetime import date

from typing import Any, Generator, Mapping, Tuple, Optional
Batch = Mapping[str, jnp.array]
OptState = Any

from source.config import HParams, Config, SystemType, OptimizerType, IntegrationOrder
from source.optimizers import TrajectoryOptimizer, get_optimizer
from source.systems import FiniteHorizonControlSystem, get_system
from source.utils import integrate, integrate_in_parallel


##############################
# Neural ODE for opt control #
##############################

def make_neural_ode(
  hp: HParams,
  cfg: Config,
  learning_rate: jnp.float32 = 0.001,
  num_training_steps: jnp.float32 = 10_001,
  use_params: Optional[str] = None,
  train_size: jnp.int32 = 100_000,
  mb_size: jnp.int32 = 128,
  order: IntegrationOrder = IntegrationOrder.CONSTANT, # only constant works
  learning_approach: str = "simple",
  save_title: Optional[str] = None
  ):

  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)
  stepsize = system.T / hp.intervals  # Segment length

  # See the same function in optimizers.py for an explanation of what this does
  # def reorganize_controls(us):
  #   state_shape = system.x_0.shape[0]
  #   control_shape = system.bounds.shape[0] - state_shape
  #   print("control shape", control_shape)
  #   midpoints_const = 2 if hp.order == IntegrationOrder.QUADRATIC else 1

  #   first = us[:-1].reshape(train_size, hp.intervals, midpoints_const*hp.controls_per_interval, control_shape)
  #   print("first", first.shape)
  #   second = us[::midpoints_const*hp.controls_per_interval][1:][:,jnp.newaxis]
  #   print("second", second.shape)
  #   return jnp.hstack([us[:-1].reshape(-1, midpoints_const*hp.controls_per_interval, control_shape),
  #                     us[::midpoints_const*hp.controls_per_interval][1:][:,jnp.newaxis]]).squeeze()

  # NOTE: for now, this will only work with single control per interval, and scalar controls.
  # TODO: make this work with arbitrary number of controls per interval
  # this will require changing the shape of the generated train_us, and reproducing
  # a version of optimizers.py's "reorganize_controls" which works over the here-new
  # dimension of number of training examples
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
    train_us = random.uniform(subkey, (train_size, hp.intervals + 1),
                              minval=u_lower, maxval=u_upper)
    if system._type == SystemType.VANDERPOL: # avoid vanderpol explosion
      train_us = train_us * 0.1
    # Integrate all the trajectories in parallel, starting from the start state,
    # and applying the randomly chosen controls, different for each trajectory
    _, train_xxs = integrate_in_parallel(system.dynamics,
                                         system.x_0[jnp.newaxis].repeat(train_size, axis=0),
                                         train_us, stepsize, hp.intervals, None, order)

    print("train xxs", train_xxs.shape)
    xs_and_us = jnp.concatenate([train_xxs, train_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    # print("returning shape", xs_and_us.shape)
    # print("training trajectories", xs_and_us[:10])
    return xs_and_us

  train_trajectories = get_many_xu_trajectories()
  print("generated training trajectories!")

  def get_minibatch(i=0):
    i = i % int(len(train_trajectories) / (mb_size + 1))
    return train_trajectories[mb_size*i:mb_size*(i+1)]

  # us is a single control trajectory
  def get_minibatch_around(us, key=random.PRNGKey(43)):
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    # NOTE: This will break with multidimensional controls
    noise = random.normal(key=key, shape=(mb_size, len(us))).squeeze()
    us = jnp.clip(us[jnp.newaxis].repeat(mb_size, axis=0).squeeze() + noise, a_min=u_lower, a_max=u_upper)
    _, train_xxs = integrate_in_parallel(system.dynamics,
                                         system.x_0[jnp.newaxis].repeat(mb_size, axis=0),
                                         us, stepsize, hp.intervals, None, order)
    xs_and_us = jnp.concatenate([train_xxs, us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    return xs_and_us

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

    # nan_to_num was necessary for van der pol (even then it didn't really train)
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

  def train_network(num_steps: int = num_training_steps, params = params, opt_state = opt_state):
    if learning_approach == "simple":
      old_loss = -1
      for step in trange(num_steps):
        if step % 1000 == 0:
          cur_loss = loss(params, get_minibatch(step))
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss
        params, opt_state = update(params, opt_state, get_minibatch(step))
    else:
      # elif learning_approach == "informed": # use informed learning approach
      #   key = random.PRNGKey(0)
      #   # Start around 0
      #   mb = get_minibatch_around(jnp.zeros(hp.intervals+1))
      #   old_loss = -1
      #   best_loss = float('inf')
      #   for step in trange(num_training_steps):
      #     # TODO: what is best strategy for selecting
      #     # controls to take randoms perturbations of?
      #     if step % 100 == 0:
      #       if optimizer.require_adj:
      #         x, u, adj = optimizer.solve()
      #       else:
      #         x, u = optimizer.solve()
      #       x_and_u, _ = ravel_pytree((x, u))
      #       inner_loss = optimizer.objective(x_and_u) # am I allowed to access this?
      #       if inner_loss < best_loss:
      #         best_loss = inner_loss
      #         # Generate trajectories around this path
      #         key, subkey = random.split(key)
      #         mb = get_minibatch_around(u, subkey)

      #     if step % 100 == 0:
      #       cur_loss = loss(params, get_minibatch(step))
      #       print(step, cur_loss)
      #       if jnp.allclose(cur_loss, old_loss):
      #         break
      #     old_loss = cur_loss
      #   params, opt_state = update(params, opt_state, get_minibatch(step))
      # else:
      #   print("Unknown learning approach")
      #   raise KeyError
      raise NotImplementedError
    
    return params, opt_state

  def apply_net(x, u):
    return net.apply(params, jnp.append(x, u))
  
  # Given a set of controls and corresponding true states, plots them,
  # and also plots the states as calculated by the NODE using those controls.
  # If get_optimal_trajectory, uses the optimal controls and corresponding
  # states, instead of those passed as arguments
  def plot_trajectory(get_optimal_trajectory: bool = False,
                      test_state: jnp.ndarray = train_trajectories[-1, :, :-1],
                      test_controls: jnp.ndarray = train_trajectories[-1, :, -1],
                      save_title: str = None):

    if get_optimal_trajectory:
      # Also get the optimal behaviour
      if optimizer.require_adj:
          x, u, adj = optimizer.solve()
          _, predicted_states = integrate(apply_net, system.x_0, u,
                                          stepsize, hp.intervals, None, order)

          system.plot_solution(x, u, adj, other_x=predicted_states, other_u=None, save_title=save_title)
      else:
          x, u = optimizer.solve()
          _, predicted_states = integrate(apply_net, system.x_0, u,
                                          stepsize, hp.intervals, None, order)

          system.plot_solution(x, u, other_x=predicted_states, other_u=None, save_title=save_title)
    else:
      _, predicted_states = integrate(apply_net, system.x_0, test_controls, stepsize, hp.intervals, None, order)

      print("test state", test_state)
      print("new state", predicted_states)
      system.plot_solution(test_state, test_controls,
                           other_x=predicted_states, other_u=None, save_title=save_title)
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

  # def train_to_different_levels(training_lengths):
  #   for length in training_lengths:

  # If use_params, then load the params from the file.
  # Otherwise, do training according to the chosen strategy.
  #
  # Strategies currently include:
  # - uniform random
  #     random exploration
  # - adaptive random (not yet implemented)
  #     every X steps, choose a new set of sample controls
  #     by sampling around the current best-performing controls

  # Finally, actually train the network and return the parameters
  if use_params:
    params = load_params(use_params)
    print("loaded params")
    print(params.keys())
  
  else:
    print("starting the multi-training")
    date_string = date.today().strftime("%Y-%m-%d")
    increment = 2000
    # small_range = [i*500 for i in range(1, 11)]
    # the_lengths = small_range + [i*10_000 for i in [1,2,3]]
    for n in [i*increment for i in range(1, 11)]:
      # Create some networks
      # params, opt_state = train_network(num_steps=increment, params=params, opt_state=opt_state)
      # pickle.dump(params, open("source/params/{}_{}_{}.p".format(hp.system.name, date_string, n), "wb"))

      # OR

      # Use the networks for testing with a tighter grid (set in config.py)
      source_params = "source/params/{}_{}_{}.p".format(hp.system.name, date_string, n)
      params = load_params(source_params)
      plot_trajectory(get_optimal_trajectory=True, save_title="source/plots/{}_{}_opt".format(hp.system.name, n))


  # plot_trajectory(get_optimal_trajectory=True)
  # plan_with_model()

  return params

def run_net(hp: HParams, cfg: Config, use_params: Optional[str] = None,
            num_training_steps: int = 1000,
            save: bool = False, learning_approach="simple",
            save_plot_title: Optional[str] = None):

  print("running net!")
  params = make_neural_ode(hp, cfg, num_training_steps=num_training_steps, use_params=use_params,
                           learning_approach=learning_approach, save_title=save_plot_title)

  if save:
    date_string = date.today().strftime("%Y-%m-%d")
    pickle.dump(params, open("source/params/{}_{}_{}.p".format(hp.system.name, date_string, num_training_steps), "wb"))

  # favorite_color = pickle.load(open("save.p", "rb"))