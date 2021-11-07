# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle as pkl

from pathlib import Path
from typing import Tuple

from myriad.config import HParams, Config, SystemType, NLPSolverType
from myriad.custom_types import Params, DParams
from myriad.defaults import learning_rates, param_guesses
from myriad.trajectory_optimizers import get_optimizer
from myriad.plotting import plot
from myriad.systems import get_name
from myriad.utils import integrate_time_independent, get_state_trajectory_and_cost, get_defect

NUM_UNROLLED = 10
# NOTE: we have a choice to make about whether we consider only
# a single trajectory at a time, or if we have a whole
# batch of trajectories (each with its own start state, and
# each with its own optimal controls) from which we sample
# at each iteration of the algorithm.


def run_endtoend(hp, cfg, num_epochs=10_000):
  if hp.system not in param_guesses:
    print("We do not currently support that kind of system for sysid. Exiting...")
    return

  data_path = f'datasets/{hp.system.name}/e2e_sysid/'
  Path(data_path).mkdir(parents=True, exist_ok=True)
  true_us_name = 'true_opt_us'
  true_xs_name = 'true_opt_xs'

  params_path = f'params/{hp.system.name}/e2e_sysid/'
  Path(params_path).mkdir(parents=True, exist_ok=True)
  params_name = f'e2e_parametric.p'

  plots_path = f'plots/{hp.system.name}/e2e_sysid/'
  Path(plots_path).mkdir(parents=True, exist_ok=True)

  guesses_path = f'intermediate_guesses/{hp.system.name}/e2e_sysid/'
  Path(guesses_path).mkdir(parents=True, exist_ok=True)

  losses_path = f'losses/{hp.system.name}/e2e_sysid/'
  Path(losses_path).mkdir(parents=True, exist_ok=True)

  true_system = hp.system()
  optimizer = get_optimizer(hp, cfg, true_system)

  # Get the true optimal controls (and state),
  # which we will try to imitate
  try:
    with open(data_path + true_us_name, 'rb') as myfile:
      true_opt_us = jnp.array(pkl.load(myfile))
    with open(data_path + true_xs_name, 'rb') as myfile:
      true_opt_xs = jnp.array(pkl.load(myfile))
    print("successfully loaded the saved optimal trajectory")
    # plt.plot(true_opt_us)
    # plt.plot(true_opt_xs)
    # plt.show()
  except Exception as e:
    print("We haven't saved the optimal trajectory for this system yet, so we'll do that now")
    true_solution = optimizer.solve()
    true_opt_us = true_solution['u']
    print("true opt us", true_opt_us.shape)
    _, true_opt_xs = integrate_time_independent(
      true_system.dynamics, true_system.x_0, true_opt_us, hp.stepsize, hp.num_steps, hp.integration_method)
    print("true opt xs", true_opt_xs.shape)

    with open(data_path + true_us_name, 'wb') as myfile:
      pkl.dump(true_opt_us, myfile)
    with open(data_path + true_xs_name, 'wb') as myfile:
      pkl.dump(true_opt_xs, myfile)

  try:
    params = pkl.load(open(params_path + params_name, 'rb'))
    print("It seems we've already trained for this system, so we'll go straight to evaluation.")
  except FileNotFoundError as e:
    print("unable to find the params, so we'll guess "
          "and then optimize and save")
    # Make a guess for our parameters
    params = param_guesses[hp.system]

    # solution_guess = optimizer.solve_with_params(params)
    # xs_and_us = solution_guess['xs_and_us']
    # lmbdas = solution_guess['lambda']

    xs_and_us = optimizer.guess
    lmbdas = jnp.zeros_like(optimizer.constraints(optimizer.guess))

    # Save the initial parameter guess so we can reset to it later
    original_xs_and_us = jnp.array(xs_and_us)
    original_lmbdas = jnp.array(lmbdas)

    # Parameter optimizer
    opt = optax.adam(hp.adam_lr)  # 1e-4
    opt_state = opt.init(params)

    # Control/state/duals optimizer
    eta_x = hp.eta_x
    eta_v = hp.eta_lmbda
    if hp.system in learning_rates:
      eta_x = learning_rates[hp.system]['eta_x']
      eta_v = learning_rates[hp.system]['eta_v']

    bounds = optimizer.bounds

    @jax.jit
    def lagrangian(xs_and_us: jnp.ndarray, lmbdas: jnp.ndarray, params: Params) -> float:
      return (optimizer.parametrized_objective(params, xs_and_us)
              + lmbdas @ optimizer.parametrized_constraints(params, xs_and_us))

    @jax.jit
    def step(x: jnp.ndarray, lmbda: jnp.ndarray, params: Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
      x_bar = jnp.clip(x - eta_x * jax.grad(lagrangian, argnums=0)(x, lmbda, params),
                       a_min=bounds[:, 0], a_max=bounds[:, 1])
      x_new = jnp.clip(x - eta_x * jax.grad(lagrangian, argnums=0)(x_bar, lmbda, params),
                       a_min=bounds[:, 0], a_max=bounds[:, 1])
      lmbda_new = lmbda + eta_v * jax.grad(lagrangian, argnums=1)(x_new, lmbda, params)
      return x_new, lmbda_new

    @jax.jit
    def step_x(x: jnp.ndarray, lmbda: jnp.ndarray, params: Params) -> jnp.ndarray:
      x_bar = jnp.clip(x - eta_x * jax.grad(lagrangian, argnums=0)(x, lmbda, params),
                       a_min=bounds[:, 0], a_max=bounds[:, 1])
      x_new = jnp.clip(x - eta_x * jax.grad(lagrangian, argnums=0)(x_bar, lmbda, params),
                       a_min=bounds[:, 0], a_max=bounds[:, 1])
      return x_new

    @jax.jit
    def step_lmbda(x: jnp.ndarray, lmbda: jnp.ndarray, params: Params) -> jnp.ndarray:
      lmbda_new = lmbda + eta_v * jax.grad(lagrangian, argnums=1)(x, lmbda, params)
      return lmbda_new

    jac_x = jax.jit(jax.jacobian(step_x, argnums=(0, 1, 2)))
    jac_lmbda = jax.jit(jax.jacobian(step_lmbda, argnums=(0, 1, 2)))

    jac_x_p = jax.jit(jax.jacobian(step_x, argnums=2))
    jac_lmbda_p = jax.jit(jax.jacobian(step_lmbda, argnums=2))

    # Update the primals and duals using the current model,
    # and also return the Jacobians of them with respect to the parameters.
    @jax.jit
    def many_steps_grad(xs_and_us: jnp.ndarray, lmbdas: jnp.ndarray, params: Params) -> DParams:
      zx = jac_x_p(xs_and_us, lmbdas, params)
      zx = jax.tree_map(lambda x: x * 0., zx)
      zlmbda = jac_lmbda_p(xs_and_us, lmbdas, params)
      zlmbda = jax.tree_map(lambda x: x * 0., zlmbda)

      @jax.jit
      def body_fun(i, vars):
        xs_and_us, lmbdas, zx, zlmbda = vars

        dx, dlmbda, dp = jac_x(xs_and_us, lmbdas, params)
        x_part = jax.tree_map(lambda el: dx @ el, zx)
        lmbda_part = jax.tree_map(lambda el: dlmbda @ el, zlmbda)
        zx = jax.tree_multimap(lambda a, b, c: a + b + c, dp, x_part, lmbda_part)
        xs_and_us = step_x(xs_and_us, lmbdas, params)

        dx, dlmbda, dp = jac_lmbda(xs_and_us, lmbdas, params)
        x_part = jax.tree_map(lambda el: dx @ el, zx)
        lmbda_part = jax.tree_map(lambda el: dlmbda @ el, zlmbda)
        zlmbda = jax.tree_multimap(lambda a, b, c: a + b + c, dp, x_part, lmbda_part)
        lmbdas = step_lmbda(xs_and_us, lmbdas, params)

        return xs_and_us, lmbdas, zx, zlmbda

      xs_and_us, lmbdas, zx, zlmbda = jax.lax.fori_loop(0, NUM_UNROLLED, body_fun, (xs_and_us, lmbdas, zx, zlmbda))

      return zx

    # Imitation loss for the optimal controls
    # def control_imitation_loss(params: Params, xs_and_us: jnp.ndarray, lmbdas: jnp.ndarray, epoch: int):
    #   for _ in range(NUM_UNROLLED):
    #     xs_and_us, lmbdas = step(xs_and_us, lmbdas, params)
    #   xs, us = optimizer.unravel(xs_and_us)
    #
    #   diff = us - true_opt_us
    #   sq_diff = diff * diff
    #   long = jnp.mean(sq_diff, axis=1)  # average all axes except time
    #   discount = (1 - 1 / (1 + jnp.exp(2 + 0.00001 * epoch))) ** jnp.arange(len(long))
    #   if hp.system in [SystemType.MOUNTAINCAR, SystemType.PENDULUM]:
    #     print("min discount", discount[-1])
    #   else:
    #     discount = 1.
    #   return jnp.mean(long * discount)

    @jax.jit
    def simple_imitation_loss(xs_and_us: jnp.ndarray, epoch):
      xs, us = optimizer.unravel(xs_and_us)

      diff = us - true_opt_us
      sq_diff = diff * diff
      long = jnp.mean(sq_diff, axis=1)  # average all axes except time
      discount = (1 - 1 / (1 + jnp.exp(2 + 0.000001 * epoch))) ** jnp.arange(len(long))
      if hp.system in [SystemType.BACTERIA, SystemType.MOUNTAINCAR, SystemType.CARTPOLE, SystemType.PENDULUM]:
        print("min discount", discount[-1])
      else:
        discount = 1.
      return jnp.mean(long * discount)

    @jax.jit
    def lookahead_update(params: Params, opt_state: optax.OptState,
                         xs_and_us: jnp.ndarray, lmbdas: jnp.ndarray, epoch: int) -> Tuple[Params, optax.OptState]:
      dx_dp = many_steps_grad(xs_and_us, lmbdas, params)
      dJ_dx = jax.grad(simple_imitation_loss)(xs_and_us, epoch)
      dJdp = jax.tree_map(lambda x: dJ_dx @ x, dx_dp)

      updates, opt_state = opt.update(dJdp, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    # Use these to record the guesses
    ts = []
    primal_guesses = []
    dual_guesses = []

    # Use this to record the losses
    imitation_losses = []

    print("starting guess of params", params)
    save_and_reset_time = 1_000
    record_things_time = 10
    for epoch in range(num_epochs):
      if epoch % save_and_reset_time == 0:
        # Check if the next params already exist (in which case we go straight to them)
        try:
          cur_params_name = f'{epoch + save_and_reset_time}e2e_parametric.p'
          params = pkl.load(open(params_path + cur_params_name, 'rb'))
          print("It seems we've already trained up to the next epoch, so we'll go straight there")
          epoch += save_and_reset_time
          continue
        except FileNotFoundError as e:
          pass

        # Record more around the very start
        record_things_time = 1

        print("saving current params")
        pkl.dump(params, open(params_path + str(epoch) + params_name, 'wb'))

        print("saving guesses so far")
        pkl.dump(ts, open(guesses_path + str(epoch) + 'ts', 'wb'))
        pkl.dump(primal_guesses, open(guesses_path + str(epoch) + 'primals', 'wb'))
        pkl.dump(dual_guesses, open(guesses_path + str(epoch) + 'duals', 'wb'))

        print("saving imitation losses")
        pkl.dump(imitation_losses, open(losses_path + str(epoch) + '_losses', 'wb'))

        print("resetting guess")
        # Reset the guess to a different random small amount
        hp.key, subkey = jax.random.split(hp.key)
        optimizer = get_optimizer(hp, cfg, true_system)
        xs_and_us = optimizer.guess
        lmbdas = original_lmbdas

      if epoch % record_things_time == 0:
        # Only have high-density recording around the start of each guess
        if epoch >= 10:
          record_things_time = 10

        # Save the current params
        ts.append(epoch)
        primal_guesses.append(np.array(xs_and_us))
        dual_guesses.append(np.array(lmbdas))

        # Save the current imitation loss
        cur_loss = simple_imitation_loss(xs_and_us, epoch)
        imitation_losses.append(cur_loss)

        print("loss", cur_loss)
        print("params", params)

      # Take step(s) with the model
      for _ in range(NUM_UNROLLED):
        xs_and_us, lmbdas = step(xs_and_us, lmbdas, params)

      # Now update to prepare for next steps
      params, opt_state = lookahead_update(params, opt_state, xs_and_us, lmbdas, epoch)

    print("Saving the final params", params)
    pkl.dump(params, open(params_path + params_name, 'wb'))

    print("Saving the final guesses")
    pkl.dump(ts, open(guesses_path + str(num_epochs - 1) + 'ts', 'wb'))
    pkl.dump(primal_guesses, open(guesses_path + str(num_epochs - 1) + 'primals', 'wb'))
    pkl.dump(dual_guesses, open(guesses_path + str(num_epochs - 1) + 'duals', 'wb'))

    print("Saving the final losses")
    pkl.dump(imitation_losses, open(losses_path + str(num_epochs - 1) + 'losses', 'wb'))

  #######################
  # Imitation loss plot #
  #######################
  b = matplotlib.get_backend()
  matplotlib.use("pgf")
  matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
  })
  plt.rcParams["figure.figsize"] = (4, 3.3)

  # Plot the imitation loss over time (params are already open, but putting this here for clarity)
  params = pkl.load(open(params_path + params_name, 'rb'))
  print("the params are", params)
  losses = pkl.load(open(losses_path + str(num_epochs - 1) + 'losses', 'rb'))
  ts = pkl.load(open(guesses_path + str(num_epochs - 1) + 'ts', 'rb'))
  primal_guesses = pkl.load(open(guesses_path + str(num_epochs - 1) + 'primals', 'rb'))

  # Plot the imitation loss over time
  plt.plot(ts, losses)
  plt.grid()
  plt.xlabel('iteration')
  plt.ylabel('imitation loss')
  plt.title("Imitation Loss")
  plt.tight_layout()
  plt.savefig(plots_path + f'imitation_loss.{cfg.file_extension}', bbox_inches='tight')
  plt.close()

  #####################
  # Control loss plot #
  #####################
  # Plot the control performance over time
  print("Plotting control performance over time")
  true_state_trajectory, optimal_cost = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, true_opt_us)

  parallel_get_state_trajectory_and_cost = jax.vmap(get_state_trajectory_and_cost, in_axes=(None, None, None, 0))
  parallel_unravel = jax.vmap(optimizer.unravel, in_axes=0)
  ar_primal_guesses = jnp.array(primal_guesses)
  _, uus = parallel_unravel(ar_primal_guesses)
  xxs, cs = parallel_get_state_trajectory_and_cost(hp, true_system, true_system.x_0, uus)

  plt.axhline(optimal_cost, color='grey', linestyle='dashed')
  plt.plot(ts, cs)
  plt.grid()
  plt.xlabel('iteration')
  plt.ylabel('cost')
  plt.title("Trajectory Cost")
  plt.tight_layout()
  plt.savefig(plots_path + f'control_performance.{cfg.file_extension}', bbox_inches='tight')
  plt.close()

  #######################
  # Final planning plot #
  #######################
  # Plot the performance of planning with the final model
  print("Plotting final planning performance")
  hp = HParams(nlpsolver=NLPSolverType.EXTRAGRADIENT)
  cfg = Config()
  true_system = hp.system()
  optimizer = get_optimizer(hp, cfg, true_system)
  learned_solution = optimizer.solve_with_params(params)
  learned_x, learned_c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, learned_solution['u'])
  learned_defect = get_defect(true_system, learned_x)

  true_x, true_c = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, true_opt_us)
  true_defect = get_defect(true_system, true_x)

  plot(hp, true_system,
       data={'x': true_opt_xs,
             'other_x': learned_x,
             'u': true_opt_us,
             'other_u': learned_solution['u'],
             'cost': true_c,
             'other_cost': learned_c,
             'defect': true_defect,
             'other_defect': learned_defect},
       labels={'x': ' (true state from controls planned with true model)',
               'other_x': ' (true state from controls planned with learned model)',
               'u': ' (planned with true model)',
               'other_u': ' (planned with learned model)'},
       styles={'x': '-',
               'other_x': 'x-',
               'u': '-',
               'other_u': 'x-'},
       widths={'x': 3,
               'other_x': 1,
               'u': 3,
               'other_u': 1},
       save_as=plots_path + f'planning_with_model.{cfg.file_extension}',
       figsize=cfg.figsize)

  #####################
  # Decision var plot #
  #####################
  # Plot showing how the guess converges to the optimal trajectory
  matplotlib.use(b)
  plt.rcParams["figure.figsize"] = (7, 5.6)
  print("Plotting convergence")
  title = get_name(hp)
  if title is not None:
    plt.suptitle(title)
  plt.suptitle(r"Intermediate Trajectories" + r"  $-$  " + title)
  plt.subplot(2, 1, 1)
  plt.grid()

  # Plot intermediate controls with transparency
  for xs in xxs:
    plt.plot(xs, color='orange', alpha=0.01)

  plt.ylabel('state (x)')
  plt.plot(true_opt_xs, label="true state from controls planned with true model", lw=3)

  # Plot the final state curve
  plt.plot(xxs[-1], 'x-', label="true state from final controls")
  plt.legend(loc='upper right')

  # Plot controls
  plt.subplot(2, 1, 2)
  plt.plot(true_opt_us, label="planned with true model", lw=3)

  # Plot intermediate controls with transparency
  for us in uus:
    plt.plot(us, color='orange', alpha=0.01)

  # Plot the final control curve
  plt.ylabel('control (u)')
  plt.xlabel('time (s)')
  plt.plot(uus[-1], 'x-', label="controls at the end of training")
  plt.legend(loc='upper right')

  plt.grid()
  plt.tight_layout()
  plt.savefig(plots_path + 'e2e_cool_plot.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
  hp, cfg = HParams(), Config()
  run_endtoend(hp, cfg)
