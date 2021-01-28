import jax.numpy as jnp
from jax import jit, grad, tree_multimap, vmap, random, nn
from jax.flatten_util import ravel_pytree
from jax import config

config.update("jax_enable_x64", True)
import haiku as hk
import optax

import fax.implicit

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import random as rd
from absl import app, flags
import pickle
from datetime import date

from typing import Any, Generator, Mapping, Tuple, Optional

Batch = Any
OptState = Any

from source.config import HParams, Config
from source.config import SystemType, OptimizerType, SamplingApproach, NLPSolverType
from source.optimizers import TrajectoryOptimizer, get_optimizer
from source.systems import FiniteHorizonControlSystem, get_system
from source.utils import integrate, integrate_in_parallel
from source.plotting import plot


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
      params_source: Optional[str] = None,
      key: jnp.ndarray = random.PRNGKey(42),
      save_every: jnp.int32 = 500,
      hidden_layers: list = [40, 40]
):
  system = get_system(hp)
  optimizer = get_optimizer(hp, cfg, system)
  num_steps = hp.intervals * hp.controls_per_interval
  stepsize = system.T / num_steps  # Segment length

  # Get the true optimal controls and corresponding trajectory
  _, true_opt_us = optimizer.solve()
  _, true_opt_xs = integrate(system.dynamics, system.x_0, true_opt_us,
                             stepsize, num_steps, None, hp.order)

  # NOTE: this might only work with scalar controls.
  # TODO: check and if necessary, extend to case with vector controls.
  # NOTE: need to behave differently if bounds are not finite
  def generate_dataset(key):
    key, subkey = random.split(key)

    # We need upper and lower bounds in order to generate random control
    # vectors for training.
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    # if system._type == SystemType.CANCER:  # CANCER usually has an infinite upper bound
    #   u_upper = 2.

    # Generate |total dataset size| control trajectories
    total_size = train_size + validation_size + test_size

    # Can't sample uniformly if the bounds are infinite,
    # so use a gaussian centered at 0
    # TODO: is there something better to do here?
    if np.isinf(u_lower) or np.isinf(u_upper):
      all_us = jnp.clip(random.normal(subkey, (total_size, num_steps + 1)),
                        a_min=u_lower, a_max=u_upper)
    else:
      all_us = random.uniform(subkey, (total_size, num_steps + 1),
                              minval=u_lower, maxval=u_upper)
    if system._type == SystemType.VANDERPOL:  # to avoid VANDERPOL dynamics explosion
      all_us = all_us * 0.1

    # Integrate all the trajectories in parallel, starting from the same start state,
    # and applying the randomly chosen controls, which are different for each trajectory
    _, all_xs = integrate_in_parallel(system.dynamics,
                                      system.x_0[jnp.newaxis].repeat(total_size, axis=0),
                                      all_us, stepsize, num_steps, None, hp.order)

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
  # be sampled from the fixed distribution anyway, to avoid getting stuck in bad states.
  def generate_dataset_around(us, key, num, dataset, spread=1):
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    # if system._type == SystemType.CANCER:  # CANCER usually has an infinite upper bound
    #   u_upper = 2.

    key, sub1, sub2, sub3 = random.split(key, 4)

    # Generate new control trajectories by adding
    # Gaussian noise to the one given
    # NOTE: This might break with multidimensional controls
    # TODO: Extend to case with vector controls.
    noise = random.normal(key=sub1, shape=(num // 2, len(us))) * spread
    new_us = jnp.clip(us + noise, a_min=u_lower, a_max=u_upper)

    # Also generate a number of random control trajectories from the fixed distribution
    random_indices = random.choice(sub2, len(dataset), shape=(num // 2,), replace=False)
    fixed_us = dataset[random_indices][:, :, 1]  # only take the controls, not the states
    assert new_us.shape == fixed_us.shape

    final_us = jnp.concatenate([new_us, fixed_us], axis=0)

    # Shuffle the control trajectories (important only if we
    # are going to sample from this dataset in contiguous chunks)
    if num > 1:
      final_us = random.permutation(sub3, final_us)

    # Generate the corresponding state trajectories
    _, train_xs = integrate_in_parallel(system.dynamics,
                                        system.x_0[jnp.newaxis].repeat(num, axis=0),
                                        final_us, stepsize, num_steps, None, hp.order)

    xs_and_us = jnp.concatenate([train_xs, final_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    assert not jnp.isnan(xs_and_us).all()

    if cfg.verbose:
      print("Generating dataset from given control trajectory:")
      print("xs shape", train_xs.shape)
      print("us shape", final_us.shape)
      print("together", xs_and_us.shape)

    return xs_and_us

  def generate_dataset_around_optimal(key, num, dataset, spread=1):
    u_lower = system.bounds[-1, 0]
    u_upper = system.bounds[-1, 1]
    # if system._type == SystemType.CANCER:  # CANCER usually has an infinite upper bound
    #   u_upper = 2.

    key, sub1 = random.split(key)

    # Generate new control trajectories by adding
    # Gaussian noise to the one given
    # NOTE: This might break with multidimensional controls
    # TODO: Extend to case with vector controls.
    noise = random.normal(key=sub1, shape=(num, len(true_opt_us))) * spread
    new_us = jnp.clip(true_opt_us + noise, a_min=u_lower, a_max=u_upper)

    # Generate the corresponding state trajectories
    _, new_xs = integrate_in_parallel(system.dynamics,
                                      system.x_0[jnp.newaxis].repeat(num, axis=0),
                                      new_us, stepsize, num_steps, None, hp.order)

    xs_and_us = jnp.concatenate([new_xs, new_us[jnp.newaxis].transpose((1, 2, 0))], axis=2)
    assert not jnp.isnan(xs_and_us).all()

    if cfg.verbose:
      print("Generating dataset from optimal controls:")
      print("xs shape", new_xs.shape)
      print("us shape", new_us.shape)
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
  # assumes dataset is of size train_size
  def yield_minibatches(dataset):
    assert len(dataset) == train_size
    tmp_dataset = np.random.permutation(dataset)
    num_minibatches = train_size // minibatch_size + (1 if train_size % minibatch_size > 0 else 0)

    for i in range(num_minibatches):
      n = np.minimum((i + 1) * minibatch_size, train_size) - i * minibatch_size
      yield tmp_dataset[i * minibatch_size: i * minibatch_size + n]

  # Find the optimal trajectory according the learned model
  def get_optimal_trajectory(params):
    opt_u = plan_with_model(params)
    _, opt_x = integrate(system.dynamics, system.x_0, opt_u,
                         stepsize, num_steps, None, hp.order)
    # assert not jnp.isnan(opt_u).all() and not jnp.isnan(opt_x).all()
    return opt_u, opt_x

  # The neural net for the neural ode: a small and simple MLP
  def net_fn(x_and_u: jnp.array) -> jnp.array:
    mlp = hk.Sequential([
      hk.Linear(hidden_layers[0]), nn.relu,
      hk.Linear(hidden_layers[1]), nn.relu,
      hk.Linear(len(system.x_0)),
    ])
    return mlp(x_and_u)  # will automatically broadcast over minibatches

  # Need to initialize things here because the later functions
  # use the nonlocal "net" object

  # Generate an initial dataset and divide it up
  key, subkey = random.split(key)
  all_data = generate_dataset(subkey)
  train_data = all_data[:train_size]
  validation_data = all_data[train_size:train_size + validation_size]
  test_data = all_data[train_size + validation_size:]
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
        minibatch: Batch
  ) -> Tuple[hk.Params, OptState]:
    grads = grad(loss)(params, minibatch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # The objective, as the agent imagines it to be (using the neural net dynamics)
  # TODO: allow objective and constraints to take custom costs and custom dynamics
  def objective_with_params(us, params):
    apply_net = lambda x, u: net.apply(params, jnp.append(x, u))

    _, xs = integrate(system.dynamics, system.x_0, us,
                      stepsize, num_steps, None, hp.order)

    # We only want the states at boundaries of shooting intervals
    xs_interval_start = xs[::hp.controls_per_interval]
    xs_and_us, unused_unravel = ravel_pytree((xs_interval_start, us))

    # Calculate the integral cost according to the learned dynamics,
    # then restory old system dynamics
    old_dynamics = system.dynamics
    system.dynamics = apply_net
    result = optimizer.objective(xs_and_us)
    system.dynamics = old_dynamics

    return result

  kkt_system = jit(grad(objective_with_params, argnums=0))

  # Solver for the root solve
  solver = lambda x_init, params: plan_with_model(params)

  def ift_update(params, opt_state):
    def J(params):
      us = fax.implicit.root_solve(
        func=kkt_system,
        init_xs=None,
        params=params,
        solver=solver,
        rev_solver=None
      )
      return divergence_from_optimal_xs(us)

    dJdp = grad(J)(params)
    # print("dJdp", dJdp)
    updates, opt_state = opt.update(dJdp, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  def get_one_step_exgd(params):
    apply_net = lambda x, u: net.apply(params, jnp.append(x, u))
    old_dynamics = system.dynamics
    system.dynamics = apply_net

    def lagrangian(xs_and_us, lmbda):
      return optimizer.objective(xs_and_us) + lmbda @ optimizer.constraints(xs_and_us)

    eta_x, eta_v = 1., 0.1
    xs_and_us = optimizer.guess
    lmbda = jnp.ones_like(optimizer.constraints(xs_and_us))

    # TODO: check if this part is slowing down computation
    # (should be able to pass parameters directly into dynamics, or optimizer, no?)
    # TODO: restructure to make code more functional

    x_bar = jnp.clip(xs_and_us - eta_x * grad(lagrangian, argnums=0)(xs_and_us, lmbda),
                     a_min=optimizer.bounds[:, 0], a_max=optimizer.bounds[:, 1])
    lmbda_bar = lmbda + eta_v * grad(lagrangian, argnums=1)(xs_and_us, lmbda)
    x_new = jnp.clip(xs_and_us - eta_x * grad(lagrangian, argnums=0)(x_bar, lmbda_bar),
                     a_min=optimizer.bounds[:, 0], a_max=optimizer.bounds[:, 1])
    lmbda_new = lmbda + eta_v * grad(lagrangian, argnums=1)(x_bar, lmbda_bar)

    system.dynamics = old_dynamics

    return x_new, lmbda_new

  def extragradient_update(params, opt_state):
    # print("starting extragradient update!")
    # def J(params):
    #   us, xs = get_optimal_trajectory(params)
    #   return divergence_from_optimal_xs(xs)
    # print("getting grad wrt params")
    # dJdp = grad(J)(params)
    # print("got grad wrt params")
    # updates, opt_state = opt.update(dJdp, opt_state)
    # new_params = optax.apply_updates(params, updates)
    # print("applied, now returning")
    # return new_params, opt_state

    def J(params):
      xs_and_us, lmbda = get_one_step_exgd(params)
      xs, us = optimizer.unravel(xs_and_us)
      return divergence_from_optimal_us(us)

    dJdp = grad(J)(params)
    # dJdu, _ = jax.jacobian(get_one_step_exgd)(params)

    updates, opt_state = opt.update(dJdp, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # NOTE: assumed 1D control
  # TODO: extend to vector control
  @jit
  def loss(cur_params: hk.Params, minibatch: Batch) -> jnp.ndarray:
    apply_net = lambda x, u: net.apply(cur_params, jnp.append(x, u))

    # Extract controls and true state trajectory
    controls = minibatch[:, :, -1]
    true_states = minibatch[:, :, :-1]

    # Use neural net to predict state trajectory
    _, predicted_states = integrate_in_parallel(apply_net,
                                                system.x_0[jnp.newaxis].repeat(len(minibatch), axis=0),
                                                controls, stepsize, num_steps, None, hp.order)

    # nan_to_num was necessary for VANDERPOL
    # (though even with this modification it still didn't train)
    # TODO: how avoid using nan_to_num, and also avoid nans?
    # true_states = jnp.nan_to_num(true_states)
    # predicted_states = jnp.nan_to_num(predicted_states)
    loss = jnp.mean((predicted_states - true_states) * (predicted_states - true_states))  # MSE
    return loss

  # TODO: jit
  # This is the "outer" loss of the problem, one of the main things we care about.
  # Another "outer" loss, which gives a more RL flavour,
  # is the integral cost of applying controls in the true dynamics,
  # and the final constraint violation (if present) when controls in the true dynamics.
  def divergence_from_optimal_us(us):
    assert len(us) == len(true_opt_us)
    return jnp.mean((us - true_opt_us) * (us - true_opt_us))  # MS

  def divergence_from_optimal_xs(us):
    # Get true state trajectory from applying "optimal" controls
    _, xs = integrate(system.dynamics, system.x_0, us,
                      stepsize, num_steps, None, hp.order)

    assert len(xs) == len(true_opt_xs)
    return jnp.mean((xs - true_opt_xs) * (xs - true_opt_xs))  # MSE

  # Load parameters saved in pickle format
  def load_params(my_pickle):
    p = None
    try:
      p = pickle.load(open(my_pickle, "rb"))
    except:
      print("Unable to find file '{}'".format(my_pickle))
      raise ValueError
    return p

  # Perform "num_minibatches" steps of minibatch gradient descent on the network
  # starting with params "params". Stores losses in the "losses" dict.
  def train_network(key, num_epochs, params, opt_state, losses={},
                    save_every=save_every, start_epoch=0):
    if not losses:
      losses = {'ts': [],
                'train_loss': [],
                'validation_loss': [],
                'loss_on_opt': [],
                'control_costs': [],
                'constraint_violation': [],
                'divergence_from_optimal_us': [],
                'divergence_from_optimal_xs': []}

    true_x_and_u_opt = jnp.concatenate([true_opt_xs, true_opt_us], axis=1)

    # # This is the outer loss
    # @jit
    # def J(params, us):
    #   del params
    #   _, xs = integrate(system.dynamics, system.x_0, us, stepsize, # true dynamics
    #                     num_steps, None, hp.order)
    #   return jnp.mean((true_opt_xs - xs)*(true_opt_xs - xs))

    def calculate_losses(params, step, cur_train_dataset=train_data):
      # Record how many training points we've used
      losses['ts'].append(start_epoch + step * train_size)

      # Calculate losses
      cur_loss = loss(params, next(yield_minibatches(cur_train_dataset)))
      losses['train_loss'].append(cur_loss)
      losses['validation_loss'].append(loss(params, next(yield_minibatches(validation_data))))
      losses['loss_on_opt'].append(loss(params, true_x_and_u_opt[jnp.newaxis]))

      # Get the optimal controls, and cost of applying them
      u = plan_with_model(params)
      _, xs = integrate(system.dynamics, system.x_0, u, stepsize,  # true dynamics
                        num_steps, None, hp.order)

      # We only want the states at boundaries of shooting intervals
      xs_interval_start = xs[::hp.controls_per_interval]
      xs_and_us, unused_unravel = ravel_pytree((xs_interval_start, u))
      losses['control_costs'].append(optimizer.objective(xs_and_us))

      # Calculate the final constraint violation, if present
      if system.x_T is not None:
        cv = system.x_T - xs[-1]
        if cfg.verbose: print("constraint violation", cv)
        losses['constraint_violation'].append(jnp.linalg.norm(cv))

      # Calculate divergences from the optimal trajectories
      losses['divergence_from_optimal_us'].append(divergence_from_optimal_us(u))
      losses['divergence_from_optimal_xs'].append(divergence_from_optimal_xs(u))

      return cur_loss

    # In this approach, training minibatches are drawn from the dataset created
    # earlier, which chose control trajectories are sampled according to a fixed distribution
    def fixed_sampling_train(params, opt_state):
      print("training on controls sampled from fixed distribution")
      old_loss = -1
      for step in trange(num_epochs):
        if step % save_every == 0:
          cur_loss = calculate_losses(params, step)  # side effect: this fills loss lists too
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
      # if system._type == SystemType.CANCER:
      #   u_upper = 2.
      if not jnp.isinf(u_lower) and not jnp.isinf(u_upper):
        u_spread = u_upper - u_lower
      else:
        u_spread = 1.

      # Initial controls are from fixed distribution
      new_dataset = train_data.copy()
      old_loss = -1
      for step in trange(num_epochs):
        if step % save_every == 0:
          # Record the losses
          cur_loss = calculate_losses(params, step,
                                      cur_train_dataset=new_dataset)  # side effect: this fills loss lists too
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss

          # Generate a new dataset around the current "optimal" controls
          key, sub = random.split(key)
          u = plan_with_model(params)
          new_dataset = generate_dataset_around(u, key=sub, num=train_size,
                                                dataset=train_data, spread=u_spread / spread_factor)

        # Descend on entire dataset, in minibatches
        for mb in yield_minibatches(new_dataset):
          params, opt_state = update(params, opt_state, mb)
      return params, opt_state

    def end_to_end_train(params, opt_state):
      print("end to end train!")
      if hp.nlpsolver != NLPSolverType.EXTRAGRADIENT:
        print("""You are attempting to run end-to-end with a non-differentiable solver.
                 While theoretically possible, we have not been able to get this to work yet.""")
        raise SystemExit
      old_loss = -1
      for step in trange(num_epochs):
        print("step", step)
        if step % save_every == 0:
          cur_loss = calculate_losses(params, step)  # side effect: this fills loss lists too
          print(step, cur_loss)
          if jnp.allclose(cur_loss, old_loss):
            break
          old_loss = cur_loss
        # In the case of extragradient, need a guess for xs, us and lambda
        if hp.nlpsolver == NLPSolverType.EXTRAGRADIENT:
          for _ in range(train_size):
            params, opt_state = extragradient_update(params, opt_state)
        else:
          for _ in range(train_size):
            # We don't care about training set, just want to have fair comparison
            # for number of training examples seen
            params, opt_state = ift_update(params, opt_state)
      return params, opt_state

    # Perform the training steps
    if hp.sampling_approach == SamplingApproach.FIXED:
      params, opt_state = fixed_sampling_train(params, opt_state)
    elif hp.sampling_approach == SamplingApproach.PLANNING:
      key, subkey = random.split(key)
      params, opt_state = planning_sampling_train(params, opt_state, subkey)
    elif hp.sampling_approach == SamplingApproach.ENDTOEND:
      params, opt_state = end_to_end_train(params, opt_state)
    else:
      print("Unknown Sampling Approach. Quitting...")
      raise ValueError

    if cfg.verbose:
      print("Trained for {} epochs on dataset of size {}".format(num_epochs, train_size))

    return params, opt_state, losses  # losses is a dict of lists which we appended to all during training

  # Plot the given control and state trajectory. Also plot the state
  # trajectory which occurs when using the neural net for dynamics.
  # If "optimal", do the same things as above but using the true
  # optimal controls and corresponding true state trajectory.
  # "extra_u" is just a way to plot an extra control trajectory.
  def plot_trajectory(params, optimal: bool = False,
                      x: jnp.ndarray = train_data[-1, :, :-1],
                      u: jnp.ndarray = train_data[-1, :, -1],
                      title=None, save_as=None):

    apply_net = lambda x, u: net.apply(params, jnp.append(x, u))  # use nonlocal net and params

    if cfg.verbose:
      print("states to plot", x.shape)
      print("controls to plot", u.shape)

    if optimal:
      x = true_opt_xs
      u = true_opt_us
      x_label = 'True state from optimal controls'
      u_label = 'True optimal controls'
    else:
      x_label = 'True state from given controls'
      u_label = 'Given controls'

    # Get states when using those controls
    _, predicted_states = integrate(apply_net, system.x_0, u,
                                    stepsize, num_steps, None, hp.order)

    # Plot
    plot(system=system,
         data={'x': x, 'u': u, 'other_x': predicted_states},
         labels={'x': x_label, 'u': u_label, 'other_x': 'Predicted state'},
         title=title, save_as=save_as)

  # First, get the optimal controls and resulting trajectory using the true system model.
  # Then, replace the model dynamics with the trained neural network,
  # and use that to find the "optimal" controls according to the NODE model.
  # Finally get the resulting true state trajectory coming from those suboptimal controls.
  def plan_with_model(cur_params=params):
    apply_net = lambda x, u: net.apply(cur_params, jnp.append(x, u))  # use nonlocal net and params

    # Replace system dynamics, but remember it to restore later
    old_dynamics = system.dynamics
    system.dynamics = apply_net

    if optimizer.require_adj:
      _, u, adj = optimizer.solve()  # _ is "dreamt" and we don't care about it
    else:
      _, u = optimizer.solve()

    # Restore system dynamics
    system.dynamics = old_dynamics

    return u.squeeze()  # this is necessary for later broadcasting

  # Plan with the model. Plot the controls from planning and corresponding true state trajectory.
  # Compare it with the true optimal controls and corresponding state trajectory.
  def plan_and_plot(params, title=None, save_as=None):
    u = plan_with_model(params)
    _, x = integrate(system.dynamics, system.x_0, u,
                     stepsize, num_steps, None, hp.order)

    plot(system=system,
         data={'x': true_opt_xs, 'u': true_opt_us, 'other_x': x, 'other_u': u},
         labels={'x': 'State trajectory from optimal controls',
                 'u': 'True optimal controls',
                 'other_x': 'State trajectory from controls given by model',
                 'other_u': 'Controls given by model'},
         title=title,
         save_as=save_as)

  def run_experiments(key, params, opt_state,
                      num_experiments=11, increment=5,
                      load_params=False, load_date=None,
                      save_weights=False, save_plots=False):
    if load_params and not load_date:
      print("Need date to load from")
      raise ValueError

    if save_weights or save_plots:
      date_string = date.today().strftime("%Y-%m-%d")

    all_losses = {}
    start_epoch = 0
    for n in [(i + 1) * increment for i in range(0, num_experiments)]:
      if load_params:
        source = "source/params/{}_{}_{}_{}.p".format(
          hp.system.name, load_date, hp.sampling_approach, n)
        params = load_params(params_source)
      else:
        key, subkey = random.split(key)
        params, opt_state, all_losses = train_network(subkey, num_epochs=increment, params=params,
                                                      opt_state=opt_state, losses=all_losses, start_epoch=start_epoch)
        start_epoch += increment * train_size
      if save_weights:
        pickle.dump(params, open("source/params/{}_{}_{}_{}.p".format(
          hp.system.name, date_string, hp.sampling_approach, n), "wb"))
      if save_plots:
        save_as = "source/plots/{}/{}_{}_{}_{}x{}".format(hp.system.name, date_string, hp.nlpsolver.name,
                                                          hp.sampling_approach.name, n, train_size)
        plot_trajectory(optimal=True, params=params,
                        title="Imitation ability on optimal trajectory after {} epochs".format(start_epoch),
                        save_as=save_as+"_im_opt")
        plot_trajectory(optimal=False, params=params,
                        title="Imitation ability on train trajectory after {} epochs".format(start_epoch),
                        save_as=save_as+"_im_rand")
        plan_and_plot(params,
                      title="Planning ability after {} epochs".format(start_epoch),
                      save_as=save_as+"_plan")

    if save_plots:
      loss_save_as = "source/plots/{}/{}_{}_{}_{}".format(hp.system.name, date_string, hp.nlpsolver.name,
                                                          hp.sampling_approach.name, train_size)
    plot_losses(all_losses,
                title="Training of {} with training size {} and {} approach".format(
                  hp.system.name, train_size, hp.sampling_approach.name),
                save_as=loss_save_as+"_loss" if save_plots else None)

    return params, all_losses

  def plot_losses(losses, title=None, save_as=None):
    ts = losses['ts']

    plt.figure(figsize=(9, 9))
    plt.subplots_adjust(left=0.115, bottom=0.05, right=0.95, top=0.91, wspace=0.32, hspace=0.31)
    if title:
      plt.suptitle(title)

    ax = plt.subplot(3, 2, 1)
    plt.plot(ts, losses['train_loss'], ".-", label="train")
    plt.plot(ts, losses['validation_loss'], ".-", label="validation")
    plt.title("imitation loss over time")
    # plt.yscale('log')
    ax.legend()

    ax = plt.subplot(3, 2, 2)
    plt.plot(ts, losses['loss_on_opt'], "o-", label=hp.sampling_approach)
    plt.title("loss over time on true optimal trajectory")
    # plt.yscale('log')
    ax.legend()

    ax = plt.subplot(3, 2, 3)
    plt.plot(ts, losses['control_costs'], ".-", label=hp.sampling_approach)
    plt.title("cost of applying \"optimal\" controls")
    ax.legend()

    if system.x_T is not None:
      ax = plt.subplot(3, 2, 4)
      plt.plot(ts, losses['constraint_violation'], ".-", label=hp.sampling_approach)
      plt.title("final constraint violation when applying those controls")
      ax.legend()

    ax = plt.subplot(3, 2, 5)
    plt.plot(ts, losses['divergence_from_optimal_us'], ".-", label=hp.sampling_approach)
    plt.title("divergence from optimal control trajectory")
    # plt.yscale('log')
    ax.legend()

    ax = plt.subplot(3, 2, 6)
    plt.plot(ts, losses['divergence_from_optimal_xs'], ".-", label=hp.sampling_approach)
    plt.title("divergence from optimal state trajectory")
    # plt.yscale('log')
    ax.legend()

    if save_as:
      plt.savefig(save_as+".pdf")
    else:
      plt.show()

  # The "make neural ode" function returns this tuple
  return params, opt_state, train_network, run_experiments


# Call this to run the the NODE
def run_net(key, hp: HParams, cfg: Config,
            params_source: Optional[str] = None,
            train_size=1_000,
            num_experiments=11,
            increment=1):
  params, opt_state, train_network, run_experiments, = make_neural_ode(hp, cfg,
                                                                       params_source=params_source,
                                                                       train_size=train_size)

  # Note that this also plots trajectories on the side
  params, losses = run_experiments(key, params, opt_state,
                                   num_experiments=num_experiments,
                                   increment=increment, save_plots=True)

  return losses
