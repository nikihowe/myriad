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
Batch = Any
OptState = Any

from source.config import HParams, Config
from source.config import SystemType, OptimizerType, SamplingApproach
from source.optimizers import TrajectoryOptimizer, get_optimizer
from source.systems import FiniteHorizonControlSystem, get_system
from source.utils import integrate, integrate_in_parallel


##############################
# Neural ODE for opt control #
##############################

# TODO: make the passing around and splitting of keys more consistent through the code
# TODO: perhaps put helper and plotting functions in a separate module
def make_neural_ode(
  hp: HParams,
  cfg: Config,
  learning_rate: jnp.float32 = 0.001,
  train_size: jnp.int32 = 5_000,
  validation_size: jnp.int32 = 1_000,
  test_size: jnp.int32 = 1_000,
  minibatch_size: jnp.int32 = 32,
  sampling_approach: SamplingApproach = SamplingApproach.UNIFORM,
  params_source: Optional[str] = None,
  plot_title: Optional[str] = None,
  key: jnp.ndarray = random.PRNGKey(42),
  save_every: jnp.int32 = 500
  ):

  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)
  num_steps = hp.intervals*hp.controls_per_interval
  stepsize = system.T / num_steps # Segment length

  # NOTE: this might only work with scalar controls.
  # TODO: check and if necessary, extend to case with vector controls.
  def generate_uniform_dataset(key):
    key, subkey = random.split(key)

    # We need upper and lower bounds in order to generate random control
    # vectors for training.
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    if system._type == SystemType.CANCER: # CANCER usually has an infinite upper bound
      u_upper = 2.

    # Generate |total dataset size| control trajectories
    total_size = train_size + validation_size + test_size
    all_us = random.uniform(subkey, (total_size, hp.intervals*hp.controls_per_interval + 1),
                            minval=u_lower, maxval=u_upper)
    if system._type == SystemType.VANDERPOL: # to avoid VANDERPOL dynamics explosion
      all_us = all_us * 0.1

    # Integrate all the trajectories in parallel, starting from the same start state,
    # and applying the randomly chosen controls, which are different for each trajectory
    _, all_xs = integrate_in_parallel(system.dynamics,
                                      system.x_0[jnp.newaxis].repeat(total_size, axis=0),
                                      all_us, stepsize, hp.intervals*hp.controls_per_interval, None, hp.order)
    
    # Stack the states and controls together
    xs_and_us = jnp.concatenate([all_xs, all_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)

    if cfg.verbose:
      print("Generating training control trajectories between bounds:")
      print("  u lower", u_lower)
      print("  u upper", u_upper)
      print("of shapes:")
      print("  xs shape", all_xs.shape)
      print("  us shape", all_us.shape)
      print("  together", xs_and_us.shape)

    return xs_and_us

  # Generate a dataset from a control trajectory by making small
  # perturbations to the controls. Make half of the dataset
  # be uniformly random anyway, to avoid getting stuck in bad states.
  def generate_dataset_around(us, key, num, spread=1):
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    if system._type == SystemType.CANCER: # CANCER usually has an infinite upper bound
      u_upper = 2.

    key, sub1, sub2, sub3 = random.split(key, 4)

    # Generate new control trajectories by adding
    # Gaussian noise to the one given
    # NOTE: This might break with multidimensional controls
    # TODO: Extend to case with vector controls.
    noise = random.normal(key=sub1, shape=(num//2, len(us))) * spread
    new_us = jnp.clip(us + noise, a_min=u_lower, a_max=u_upper)
    # Also generate a number of uniform random control trajectories
    uniform_us = random.uniform(key=sub2, shape=(num//2, len(us)),
                                minval=u_lower, maxval=u_upper)
    final_us = jnp.concatenate([new_us, uniform_us], axis=0)

    # Shuffle the control trajectories (important only if we
    # are going to sample from this dataset in contiguous chunks)
    if num > 1:
      final_us = random.permutation(sub3, final_us)

    # Generate the corresponding state trajectories
    _, train_xs = integrate_in_parallel(system.dynamics,
                                        system.x_0[jnp.newaxis].repeat(num, axis=0),
                                        final_us, stepsize, hp.intervals*hp.controls_per_interval, None, hp.order)
    
    xs_and_us = jnp.concatenate([train_xs, final_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    assert not jnp.isnan(xs_and_us).all()

    if cfg.verbose:
      print("Generating dataset from given control trajectory:")
      print("xs shape", train_xs.shape)
      print("us shape", final_us.shape)
      print("together", xs_and_us.shape)

    return xs_and_us

  # Various ways to get a minibatch from a dataset
  # def get_minibatch(dataset, i=0, approach="full_batch", key=None):
  #   if approach == "numpy": # fast and random, but not jax
  #     random_indices = np.random.choice(len(dataset), minibatch_size, replace=False)
  #     return dataset[random_indices]
  #   elif approach == "full_batch":
  #     return dataset
  #   elif approach == "deterministic": # really fast, but no randomness
  #     i = i % int(len(dataset) / (minibatch_size + 1))
  #     return dataset[minibatch_size*i:minibatch_size*(i+1)]
  #   elif approach == "jax" and key is not None: # pure jax, but very slow
  #     random_indices = random.choice(key, len(dataset), shape=(minibatch_size,), replace=False)
  #     return dataset[random_indices]
  #   else:
  #     print("Unknown approach. Please choose among \{numpy, deterministic, jax\}")
  #     raise ValueError

  # TODO: make it use jax instead (can it still be fast then?)
  def yield_minibatches(dataset):
    tmp_train_data = np.random.permutation(train_data)
    num_minibatches = train_size // minibatch_size + (1 if train_size % minibatch_size > 0 else 0)

    for i in range(num_minibatches):
      n = np.minimum((i+1) * minibatch_size, train_size) - i*minibatch_size
      yield tmp_train_data[i*minibatch_size : i*minibatch_size + n]

  # Find the optimal trajectory according the learned model
  def get_optimal_trajectory(params):
    opt_u = plan_with_model(params)
    _, opt_x = integrate(system.dynamics, system.x_0, opt_u,
                         stepsize, hp.intervals*hp.controls_per_interval, None, hp.order)
    xs_and_us = jnp.concatenate([opt_x, opt_u[:,jnp.newaxis]], axis=1)
    assert not jnp.isnan(xs_and_us).all()
    return xs_and_us

  # The neural net for the neural ode: a small and simple MLP
  def net_fn(x_and_u: jnp.array) -> jnp.array:
    mlp = hk.Sequential([
        hk.Linear(40), nn.relu,
        hk.Linear(40), nn.relu,
        hk.Linear(len(system.x_0)),
    ])
    return mlp(x_and_u) # will automatically broadcast over minibatches

  # Need to initialize things here because the later functions
  # use the nonlocal "net" object

  # Generate an initial dataset and divide it up
  key, subkey = random.split(key)
  all_data = generate_uniform_dataset(subkey)
  train_data = all_data[:train_size]
  validation_data = all_data[train_size:train_size+validation_size]
  test_data = all_data[train_size+validation_size:]
  if cfg.verbose:
    print("Generated training trajectories of shape", train_data.shape)
    print("Generated validation trajectories of shape", validation_data.shape)
    print("Generated test trajectories of shape", test_data.shape)

  # Initialize the parameters and optimizer state
  net = hk.without_apply_rng(hk.transform(net_fn))
  mb = next(yield_minibatches(train_data))
  key, subkey = random.split(key)
  params = net.init(subkey, mb[-1])
  opt = optax.adam(learning_rate)
  opt_state = opt.init(params)
  if cfg.verbose:
    print("minibatches are of shape", mb.shape)
    print("initialized network weights")

  # Gradient descent on the loss function in scope
  @jit
  def update(
      params: hk.Params,
      opt_state: OptState,
      minibatch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    grads = grad(loss)(params, minibatch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # NOTE: assumed 1D control
  # TODO: extend to vector control
  @jit
  def loss(cur_params: hk.Params, minibatch: Batch) -> jnp.ndarray:
    apply_net = lambda x, u : net.apply(cur_params, jnp.append(x, u))

    # Extract controls and true state trajectory
    controls = minibatch[:, :, -1]
    true_states = minibatch[:, :, :-1]

    # Use neural net to predict state trajectory
    _, predicted_states = integrate_in_parallel(apply_net,
                                                system.x_0[jnp.newaxis].repeat(len(minibatch), axis=0),
                                                controls, stepsize, hp.intervals*hp.controls_per_interval, None, hp.order)

    
    # nan_to_num was necessary for VANDERPOL
    # (though even with this modification it still didn't train)
    # TODO: I wonder if there is a way to avoid the ugly nan_to_num
    # true_states = jnp.nan_to_num(true_states)
    # predicted_states = jnp.nan_to_num(predicted_states)
    loss = jnp.mean((predicted_states - true_states)*(predicted_states - true_states)) # MSE
    return loss

  # Load parameters saved in pickle format
  def load_params(my_pickle):
    p = None
    try:
      p = pickle.load(open(my_pickle, "rb"))
    except:
      print("Unable to find file '{}'".format(my_pickle))
      raise SystemExit
    return p

  # Perform "num_minibatches" steps of minibatch gradient descent on the network
  # starting with params "params". Stores losses in the "losses" dict.
  def train_network(key, num_epochs, params, opt_state, losses={}, save_every=save_every, start_epoch=0):

    if not losses:
      losses = {'ts':[],
                'train_loss':[], 
                'validation_loss':[], 
                'loss_on_opt':[], 
                'control_costs':[],
                'constraint_violation':[]}

    # Get the optimal u (according to the model which uses this NODE for dynamics)
    # and the corresponding state trajectory that occurs when they are applied
    # following the true environment dynamics
    x_and_u_opt = get_optimal_trajectory(params)

    def calculate_losses(params, step):
      # Record how many training points we've used
      losses['ts'].append(start_epoch + step * train_size)

      # Calculate losses
      cur_loss = loss(params, next(yield_minibatches(train_data)))
      losses['train_loss'].append(cur_loss)
      losses['validation_loss'].append(loss(params, next(yield_minibatches(validation_data))))
      losses['loss_on_opt'].append(loss(params, x_and_u_opt[jnp.newaxis]))

      # Get the optimal controls, and cost of applying them
      u = plan_with_model(params)
      _, xs = integrate(system.dynamics, system.x_0, u, stepsize, # true dynamics
                        num_steps, None, hp.order)

      # We only want the states at boundaries of shooting intervals
      xs_interval_start = xs[::hp.controls_per_interval]
      xs_and_us, unused_unravel = ravel_pytree((xs_interval_start, u))
      losses['control_costs'].append(optimizer.objective(xs_and_us))

      # Calculate the final constraint violation, if present
      if system.x_T is not None:
        cv = system.x_T - xs[-1]
        print("constraint violation", cv)
        losses['constraint_violation'].append(jnp.linalg.norm(cv))

      return cur_loss

    # In this approach, training minibatches are drawn from the dataset created
    # earlier, which chose control trajectories uniformly at random
    def uniform_sampling_train(params, opt_state):
      print("sampling from uniform controls")
      old_loss = -1
      for step in trange(num_epochs):
        if step % save_every == 0:
          cur_loss = calculate_losses(params, step) # side effect: this fills loss lists too
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss
        # Descend on entire dataset, in minibatches
        for mb in yield_minibatches(train_data):
          params, opt_state = update(params, opt_state, mb)
      return params, opt_state

    # In this approach, training minibatches are drawn from a dataset repeatedly
    # created around the currently-optimal control sequence ("optimal" as according to
    # when we use the NODE as dynamics model).
    # "spread_factor" describes how tightly you want to sample around those controls
    # (bigger spread factor === tighter sampling)
    def planning_sampling_train(params, opt_state, key, spread_factor=4):
      print("sampling around planned controls")
      # Start around average
      u_lower = system.bounds[-1, 0]
      u_upper = system.bounds[-1, 1]
      if system._type == SystemType.CANCER:
        u_upper = 2.
      u_spread = u_upper - u_lower

      # Initial controls are uniform at random
      mb = train_data
      old_loss = -1
      for step in trange(num_epochs):
        if step % save_every == 0:
          cur_loss = calculate_losses(params, step) # side effect: this fills loss lists too
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss

          # Generate a new dataset around the current "optimal" controls
          u = plan_with_model(params)
          new_dataset = generate_dataset_around(u, key=key, num=train_size, spread=u_spread/spread_factor)

        # Descend on entire dataset, in minibatches
        for mb in yield_minibatches(new_dataset):
          params, opt_state = update(params, opt_state, mb)
      return params, opt_state

    # Perform the training steps
    if sampling_approach == SamplingApproach.UNIFORM:
      params, opt_state = uniform_sampling_train(params, opt_state)
    elif sampling_approach == SamplingApproach.PLANNING:
      key, subkey = random.split(key)
      params, opt_state = planning_sampling_train(params, opt_state, subkey)

    if cfg.verbose:
      print("Trained for {} epochs on dataset of size {}".format(num_epochs, train_size))

    return params, opt_state, losses # losses is a dict of lists which we appended to all during training

  # Plot the given control and state trajectory. Also plot the state
  # trajectory which occurs when using the neural net for dynamics.
  # If "optimal", do the same things as above but using the true
  # optimal controls and corresponding true state trajectory.
  # "extra_u" is just a way to plot an extra control trajectory.
  def plot_trajectory(optimal: bool = False,
                      x: jnp.ndarray = train_data[-1, :, :-1],
                      u: jnp.ndarray = train_data[-1, :, -1],
                      extra_u: Optional[jnp.ndarray] = None,
                      plot_title: str = None):

    apply_net = lambda x, u : net.apply(params, jnp.append(x, u)) # use nonlocal net and params

    if cfg.verbose:
      print("states to plot", x.shape)
      print("controls to plot", u.shape)

    # Get optimal controls, if we so desire
    if optimal:
      if optimizer.require_adj:
          x, u, adj = optimizer.solve()
      else:
          x, u = optimizer.solve()

    # Get states when using those controls
    _, predicted_states = integrate(apply_net, system.x_0, u,
                                    stepsize, hp.intervals*hp.controls_per_interval, None, hp.order)

    # Plot
    if optimizer.require_adj:
        system.plot_solution(x, u, adj, other_x=predicted_states, other_u=extra_u, plot_title=plot_title)
    else:
      system.plot_solution(x, u, other_x=predicted_states, other_u=extra_u, plot_title=plot_title)

  # First, get the optimal controls and resulting trajectory using the true system model.
  # Then, replace the model dynamics with the trained neural network,
  # and use that to find the "optimal" controls according to the NODE model.
  # Finally get the resulting true state trajectory coming from those suboptimal controls.
  def plan_with_model(cur_params=params):
    apply_net = lambda x, u : net.apply(cur_params, jnp.append(x, u)) # use nonlocal net and params

    # Replace system dynamics, but remember it to restore later
    old_dynamics = system.dynamics
    system.dynamics = apply_net
    new_optimizer = get_optimizer(hp, cfg, system)

    # Plan with NODE model
    if new_optimizer.require_adj:
      _, u, adj = new_optimizer.solve() # _ is "dreamt" and we don't care about it
    else:
      _, u = new_optimizer.solve()

    # Restore system dynamics
    system.dynamics = old_dynamics

    return u.squeeze() # this is necessary for later broadcasting

  def run_experiments(key, params, opt_state,
                      num_experiments=11, increment=2000,
                      load_params=False, load_date=None,
                      save_weights=False, save_plots=False):
    if load_params and not load_date:
      print("Need date to load from")
      raise ValueError

    if save_weights or save_plots:
      date_string = date.today().strftime("%Y-%m-%d")

    all_losses = {}
    start_epoch = 0
    for n in [(i+1)*increment for i in range(0, num_experiments)]:
      if load_params:
        source = "source/params/{}_{}_{}_{}.p".format(
          hp.system.name, load_date, sampling_approach, n)
        params = load_params(params_source)
      else:
        key, subkey = random.split(key)
        params, opt_state, all_losses = train_network(subkey, num_epochs=increment, params=params,
                                                      opt_state=opt_state, losses=all_losses, start_epoch=start_epoch)
        start_epoch += increment*train_size
      if save_weights:
        pickle.dump(params, open("source/params/{}_{}_{}_{}.p".format(
          hp.system.name, date_string, sampling_approach, n), "wb"))
      if save_plots:
        us = plan_with_model(params)
        plot_trajectory(optimal=True, extra_u=us,
                        plot_title="source/plots/{}_{}_{}_{}_opt".format(
                          hp.system.name, date_string, sampling_approach, n))

    return all_losses

  # The "make neural ode" function returns this tuple
  return params, opt_state, train_network, run_experiments

# Call this to run the the NODE
def run_net(key, hp: HParams, cfg: Config, params_source: Optional[str] = None,
            save: bool = False, sampling_approach=SamplingApproach.UNIFORM,
            save_plot_title: Optional[str] = None, train_size=5_000):

  params, opt_state, train_network, run_experiments = make_neural_ode(hp, cfg, params_source=params_source,
                                                                      sampling_approach=sampling_approach, plot_title=save_plot_title)

  losses = run_experiments(key, params, opt_state, num_experiments=3, increment=10)
  return losses