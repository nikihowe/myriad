# (c) 2021 Nikolaus Howe
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle as pkl

from pathlib import Path
from typing import Tuple, Optional

from myriad.config import HParams, Config, SystemType, NLPSolverType
from myriad.defaults import learning_rates, param_guesses
from myriad.neural_ode.create_node import NeuralODE
from myriad.custom_types import Control, Params, State, Timestep, DParams
from myriad.trajectory_optimizers import get_optimizer
from myriad.plotting import plot
from myriad.systems.node_system import NodeSystem
from myriad.systems import get_name
from myriad.utils import integrate_time_independent, get_state_trajectory_and_cost, get_defect


def run_node_endtoend(hp, cfg, num_epochs=10_000, load_specific_epoch_params=None):
  if hp.system not in param_guesses:
    print("We do not currently support that kind of system for sysid. Exiting...")
    return

  data_path = f'datasets/{hp.system.name}/node_e2e_sysid/'
  Path(data_path).mkdir(parents=True, exist_ok=True)
  true_us_name = 'true_opt_us'
  true_xs_name = 'true_opt_xs'

  params_path = f'params/{hp.system.name}/node_e2e_sysid/'
  Path(params_path).mkdir(parents=True, exist_ok=True)
  params_name = f'node_e2e.p'

  plots_path = f'plots/{hp.system.name}/node_e2e_sysid/'
  Path(plots_path).mkdir(parents=True, exist_ok=True)

  guesses_path = f'intermediate_guesses/{hp.system.name}/node_e2e_sysid/'
  Path(guesses_path).mkdir(parents=True, exist_ok=True)

  losses_path = f'losses/{hp.system.name}/node_e2e_sysid/'
  Path(losses_path).mkdir(parents=True, exist_ok=True)

  node = NeuralODE(hp, cfg, mle=False)
  true_system = hp.system()  # use the default params here
  true_optimizer = get_optimizer(hp, cfg, true_system)
  node_system = NodeSystem(node=node, true_system=true_system)
  node_optimizer = get_optimizer(hp, cfg, node_system)

  # Get the true optimal controls (and state),
  # which we will try to imitate
  try:
    with open(data_path + true_us_name, 'rb') as myfile:
      true_opt_us = jnp.array(pkl.load(myfile))
    with open(data_path + true_xs_name, 'rb') as myfile:
      true_opt_xs = jnp.array(pkl.load(myfile))
    print("successfully loaded the saved optimal trajectory")
  except Exception as e:
    print("We haven't saved the optimal trajectory for this system yet, so we'll do that now")
    true_solution = true_optimizer.solve()
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
    node.load_params(params_path + params_name)
    print("It seems we've already trained for this system, so we'll go straight to evaluation.")
  except FileNotFoundError as e:
    print("unable to find the params, so we'll guess "
          "and then optimize and save")

    xs_and_us = true_optimizer.guess
    lmbdas = jnp.zeros_like(true_optimizer.constraints(true_optimizer.guess))

    # As a sanity check, use the true optimal controls and see if we diverge from them
    # opt_xs_and_us = pkl.load(open('bleble', 'rb'))
    # print('xs_and_us', xs_and_us.shape)

    # Save these so we can reset them later
    original_xs_and_us = jnp.array(xs_and_us)
    original_lmbdas = jnp.array(lmbdas)

    # Parameter optimization
    opt = optax.adam(1e-4)
    opt_state = opt.init(node.params)

    # Control/state/duals optimizer
    eta_x = hp.eta_x
    eta_v = hp.eta_lmbda
    if hp.system in learning_rates:
      eta_x = learning_rates[hp.system]['eta_x']
      eta_v = learning_rates[hp.system]['eta_v']

    bounds = true_optimizer.bounds

    @jax.jit
    def lagrangian(xs_and_us: jnp.ndarray, lmbdas: jnp.ndarray, params: Params) -> float:
      return (node_optimizer.parametrized_objective(params, xs_and_us)
              + lmbdas @ node_optimizer.parametrized_constraints(params, xs_and_us))

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
        x_part = jax.tree_map(lambda el: jnp.tensordot(dx, el, axes=(1, 0)), zx)
        lmbda_part = jax.tree_map(lambda el: jnp.tensordot(dlmbda, el, axes=(1, 0)), zlmbda)
        zx = jax.tree_multimap(lambda a, b, c: a + b + c, dp, x_part, lmbda_part)
        xs_and_us = step_x(xs_and_us, lmbdas, params)

        dx, dlmbda, dp = jac_lmbda(xs_and_us, lmbdas, params)
        x_part = jax.tree_map(lambda el: jnp.tensordot(dx, el, axes=(1, 0)), zx)
        lmbda_part = jax.tree_map(lambda el: jnp.tensordot(dlmbda, el, axes=(1, 0)), zlmbda)
        zlmbda = jax.tree_multimap(lambda a, b, c: a + b + c, dp, x_part, lmbda_part)
        lmbdas = step_lmbda(xs_and_us, lmbdas, params)

        return xs_and_us, lmbdas, zx, zlmbda

      xs_and_us, lmbdas, zx, zlmbda = jax.lax.fori_loop(0, hp.num_unrolled, body_fun, (xs_and_us, lmbdas, zx, zlmbda))

      return zx

    # @jax.jit
    # def control_imitation_loss(params: Params, init_xs_and_us: jnp.ndarray, init_lmbdas: jnp.ndarray):
    #   xs_and_us_new, lmbda_new = step(init_xs_and_us, init_lmbdas, params)
    #   xs, us = true_optimizer.unravel(xs_and_us_new)
    #   return jnp.mean((us - true_opt_us) ** 2)  # same loss as "Diff. MPC"

    @jax.jit
    def simple_imitation_loss(xs_and_us: jnp.ndarray, epoch: int):
      xs, us = true_optimizer.unravel(xs_and_us)
      diff = us - true_opt_us
      sq_diff = diff * diff
      long = jnp.mean(sq_diff, axis=1)
      discount = (1 - 1 / (1 + jnp.exp(2 + 0.000001 * epoch))) ** jnp.arange(len(long))
      if hp.system in []:
        print("min discount", discount[-1])
      else:
        discount = 1.
      return jnp.mean(long * discount)

    @jax.jit
    def lookahead_update(params: Params, opt_state: optax.OptState, xs_and_us: jnp.ndarray,
                         lmbdas: jnp.ndarray, epoch: int) -> Tuple[Params, optax.OptState]:
      dloop_dp = many_steps_grad(xs_and_us, lmbdas, params)
      dx_dloop = jax.grad(simple_imitation_loss)(xs_and_us, epoch)
      dJdp = jax.tree_map(lambda x: jnp.tensordot(dx_dloop, x, axes=(0, 0)), dloop_dp)

      updates, opt_state = opt.update(dJdp, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    # Use to record the guesses
    ts = []
    primal_guesses = []
    dual_guesses = []

    # Use to record the losses
    imitation_losses = []

    # print("true params", true_params)
    # print("starting guess of params", node.params)
    # u_lower = true_system.bounds[hp.state_size:, 0]
    # u_upper = true_system.bounds[hp.state_size:, 1]

    record_things_time = 10
    save_and_reset_time = 1000
    for epoch in range(num_epochs):
      if epoch % save_and_reset_time == 0:
        # Record more around the very start
        record_things_time = 1

        print("saving current params")
        pkl.dump(node.params, open(params_path + str(epoch) + params_name, 'wb'))

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

        xs, cur_us = true_optimizer.unravel(xs_and_us)
        plt.ion()
        fig = plt.figure()
        # if a_plt is None:
        ax1 = fig.add_subplot(211)
        a_plt = ax1.plot(true_opt_xs, label="true opt xs")
        b_plt = ax1.plot(xs, label="xs from given controls")
        plt.legend()
        ax2 = fig.add_subplot(212)
        c_plt = ax2.plot(true_opt_us, label="true opt us")
        d_plt = ax2.plot(cur_us, label="current us")
        plt.legend()
        # plt.show()
        # else:
        #   b_plt[0].set_ydata(predicted_states[:, 0])
        #   b_plt[1].set_ydata(predicted_states[:, 1])
        # b_plt = ax1.plot(np.sin(np.arange(epoch, epoch+10)), label="predicted xs")

        plt.savefig(f"{plots_path}progress_epoch_{epoch}.png")
        plt.close()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save the current params
        ts.append(epoch)
        primal_guesses.append(np.array(xs_and_us))
        dual_guesses.append(np.array(lmbdas))

        # Save the current imitation loss
        cur_loss = simple_imitation_loss(xs_and_us, epoch)
        imitation_losses.append(cur_loss)

        print(epoch, "loss", cur_loss)

      # Take step(s) with the model
      for _ in range(hp.num_unrolled):
        xs_and_us, lmbdas = step(xs_and_us, lmbdas, node.params)

      # Use the new technique for updating
      node.params, opt_state = lookahead_update(node.params, opt_state, xs_and_us, lmbdas, epoch)

    print("Saving the final params", node.params)
    pkl.dump(node.params, open(params_path + params_name, 'wb'))

    print("Saving the final guesses")
    pkl.dump(ts, open(guesses_path + str(num_epochs - 1) + 'ts', 'wb'))
    pkl.dump(primal_guesses, open(guesses_path + str(num_epochs - 1) + 'primals', 'wb'))
    pkl.dump(dual_guesses, open(guesses_path + str(num_epochs - 1) + 'duals', 'wb'))

    print("Saving the final losses")
    pkl.dump(imitation_losses, open(losses_path + str(num_epochs - 1) + 'losses', 'wb'))

  if cfg.plot:
    # Plot the imitation loss over time  # params are already open, but putting this here for clarity

    if load_specific_epoch_params is not None:
      node.load_params(params_path + str(load_specific_epoch_params) + params_name)
      losses = pkl.load(open(losses_path + str(load_specific_epoch_params) + '_losses', 'rb'))
      ts = pkl.load(open(guesses_path + str(load_specific_epoch_params) + 'ts', 'rb'))
      primal_guesses = pkl.load(open(guesses_path + str(load_specific_epoch_params) + 'primals', 'rb'))
    else:
      node.load_params(params_path + params_name)
      losses = pkl.load(open(losses_path + str(num_epochs - 1) + 'losses', 'rb'))
      ts = pkl.load(open(guesses_path + str(num_epochs - 1) + 'ts', 'rb'))
      primal_guesses = pkl.load(open(guesses_path + str(num_epochs - 1) + 'primals', 'rb'))

    # Check the lengths
    if len(losses) % 100 == 0: # 10000:
      print("clipping losses")
      losses = losses[:-1]
    if len(ts) % 100 == 0: # == 10000:
      print("clipping ts")
      ts = ts[:-1]
    if len(primal_guesses) % 100 == 0: # == 10000:
      print("clipping primal guesses")
      primal_guesses = primal_guesses[:-1]

    # assert len(losses) == 999
    # assert len(ts) == 999
    # assert len(primal_guesses) == 999

    ##################
    # Imitation loss #
    ##################
    b = matplotlib.get_backend()
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
      "pgf.texsystem": "pdflatex",
      'font.family': 'serif',
      'text.usetex': True,
      'pgf.rcfonts': False,
    })
    plt.rcParams["figure.figsize"] = (4, 3.3)

    # Plot the imitation loss over time
    plt.plot(ts, losses)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('imitation loss')
    plt.title("Imitation Loss")
    plt.tight_layout()
    plt.savefig(plots_path + f'imitation_loss.{cfg.file_extension}', bbox_inches='tight')
    plt.close()

    #######################
    # Control performance #
    #######################
    true_state_trajectory, optimal_cost = get_state_trajectory_and_cost(hp, true_system, true_system.x_0, true_opt_us)

    # Plot the control performance over time
    print("Plotting control performance over time")
    parallel_get_state_trajectory_and_cost = jax.vmap(get_state_trajectory_and_cost, in_axes=(None, None, None, 0))
    parallel_unravel = jax.vmap(true_optimizer.unravel, in_axes=0)
    ar_primal_guesses = np.array(primal_guesses)
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

    # Save the plot
    # plt.savefig(f"{plots_path + plots_name}_epoch_{epoch}.png")
    # plt.close()

    # Plot the performance of planning with the final model
    print("Plotting final planning performance")
    hp = HParams(nlpsolver=NLPSolverType.EXTRAGRADIENT)
    cfg = Config()
    node_optimizer = get_optimizer(hp, cfg, node_system)
    learned_solution = node_optimizer.solve_with_params(node.params)

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
    # plt.suptitle("Intermediate Trajectories")
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
    plt.savefig(plots_path + 'node_e2e_cool_plot.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
  hp = HParams()
  cfg = Config()
  run_node_endtoend(hp, cfg)
