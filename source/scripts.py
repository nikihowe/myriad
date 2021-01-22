# put this in run.py in order to get the four-plot ones (need to also call run_experiment in opt_control_neural_ode)

key1, key2 = jax.random.split(jax.random.PRNGKey(42))
losses_simple = run_net(key1, hp, cfg, sampling_approach=SamplingApproach.UNIFORM)
losses_informed = run_net(key2, hp, cfg, sampling_approach=SamplingApproach.PLANNING)

ts = losses_simple['ts']

plt.figure(figsize=(10, 9))

ax = plt.subplot(2, 2, 1)
plt.plot(ts, losses_simple['train_loss'], ".-", label="train")
plt.plot(ts, losses_simple['validation_loss'], ".-", label="validation")
plt.title("\"simple\" approach's loss over time")
plt.yscale('log')
ax.set_ylim([1e-10, 1e3])
ax.legend()

ax = plt.subplot(2, 2, 2)
plt.plot(ts, losses_informed['train_loss'], ".-", label="train")
plt.plot(ts, losses_informed['validation_loss'], ".-", label="validation")
plt.title("\"informed\" approach's loss over time")
plt.yscale('log')
ax.set_ylim([1e-10, 1e3])
ax.legend()

# ax = plt.subplot(2, 2, 3)
# plt.plot(losses_simple['loss_on_opt'], "o-", label="simple")
# plt.plot(losses_informed['loss_on_opt'], "o-", label="informed")
# plt.title("loss over time on true optimal trajectory")
# plt.yscale('log')
# ax.legend()

ax = plt.subplot(2, 2, 3)
plt.plot(ts, losses_simple['control_costs'], ".-", label="simple")
plt.plot(ts, losses_informed['control_costs'], ".-", label="informed")
plt.title("cost of applying \"optimal\" controls")
# plt.yscale('log')
ax.legend()

ax = plt.subplot(2, 2, 4)
plt.plot(ts, losses_simple['constraint_violation'], ".-", label="simple")
plt.plot(ts, losses_informed['constraint_violation'], ".-", label="informed")
plt.title("final constraint violation when applying those controls")
# plt.yscale('log')
ax.legend()

plt.show()

# -------------------------------------------
# Script to compare training for different amounts of time
# put it in run.py

date_string = date.today().strftime("%Y-%m-%d")

# Train for different amounts of time
for n in [i*10_000 for i in range(1, 11)]:
  print("num_training_steps", n)
  run_net(hp, cfg, num_training_steps=n,
          save_plot_title="{}_{}_{}".format(hp.system.name, date_string, n))

# Test the quality of the different trainings
from datetime import date
for n in [i*10_000 for i in range(1, 11)]:
  date_string = date.today().strftime("%Y-%m-%d")
  name = "source/params/{}_{}_{}.p".format(hp.system.name, n, date_string)
  run_net(hp, cfg, use_params=name,
          save_plot_title="{}_{}".format(hp.system.name, n))
  break

# --------------------------------------
# Script for running the standard trajectory optimization

# put this in run.py
optimizer = get_optimizer(hp, cfg, system)
if optimizer.require_adj:
  x, u, adj = optimizer.solve()
else:
  x, u = optimizer.solve()

num_steps = hp.intervals*hp.controls_per_interval
stepsize = system.T / num_steps
_, opt_x = integrate(system.dynamics, system.x_0, u,
                     stepsize, num_steps, None, hp.order)

if cfg.plot_results:
  if optimizer.require_adj:
    system.plot_solution(x, u, adj, opt_x)
  else:
    system.plot_solution(x, u, other_x=opt_x)

xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
if hp.optimizer != OptimizerType.FBSM:
  print("control cost", optimizer.objective(xs_and_us))
  print('constraint_violations', jnp.linalg.norm(optimizer.constraints(xs_and_us)))
raise SystemExit

# --------------------------------------------
# Script for running the dataset ablation test

# put this in run.py
key = jax.random.PRNGKey(42)
for train_size in [i*8 for i in range(1,11)]:
  key, subkey = jax.random.split(key)
  losses_simple = run_net(subkey, hp, cfg, sampling_approach=SamplingApproach.UNIFORM, train_size=train_size)

  ts = losses_simple['ts']

  num_subplots = 3 if system.x_T is not None else 2
  plot_height = 9 if system.x_T is not None else 7

  fig = plt.figure(figsize=(8, 9))
  fig.suptitle("Train set of size {}".format(train_size))

  ax = plt.subplot(num_subplots, 1, 1)
  plt.plot(ts, losses_simple['train_loss'], ".-", label="train")
  plt.plot(ts, losses_simple['validation_loss'], ".-", label="validation")
  plt.title("\"simple\" approach's loss over time")
  plt.yscale('log')
  ax.set_ylim([1e-7, 2])
  ax.legend()

  ax = plt.subplot(num_subplots, 1, 2)
  plt.plot(ts, losses_simple['control_costs'], ".-")
  plt.title("cost of applying \"optimal\" controls")

  if system.x_T is None:
    plt.xlabel("# train trajectories seen (including repetitions)")

  if system.x_T is not None:
    ax = plt.subplot(num_subplots, 1, 3)
    plt.plot(ts, losses_simple['constraint_violation'], ".-")
    plt.title("constraint violation when applying those controls")
    plt.xlabel("# train trajectories seen (including repetitions)")

  fig.tight_layout()
  plt.savefig("source/plots/dset_ablation_{}_{}.pdf".format(hp.system.name, train_size)) 

# and this in opt_control_neural_ode.py

def run_net(key, hp: HParams, cfg: Config, params_source: Optional[str] = None,
          save: bool = False, sampling_approach=SamplingApproach.UNIFORM,
          save_plot_title: Optional[str] = None, train_size=100_000):

params, opt_state, train_network = make_neural_ode(hp, cfg, params_source=params_source,
                                            sampling_approach=sampling_approach,
                                            train_size=train_size,
                                            minibatch_size=8,
                                            plot_title=save_plot_title,
                                            save_every=500)

params, opt_state, losses = train_network(key, num_epochs=5001, params=params, opt_state=opt_state)

return losses

# ---------------------------
# hyperparameter tuning for extragradient

  from hyperopt import hp as h
  from hyperopt import fmin, tpe, pyll, STATUS_OK, STATUS_FAIL
  import pprint
  pp = pprint.PrettyPrinter(indent=4, width=100)

  # Parameter search for extragradient
  # define an objective function
  def f(space):
    extra_options = {
      'maxiter': space['maxiter'],
      'eta_x': 10**space['eta_x_exp'],
      'eta_v': 10**space['eta_v_exp'],
      'atol': 10**space['atol_exp'],
    }

    x, u = optimizer.solve(extra_options)

    print("xs", x.shape)
    print("us", u.shape)

    xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
    obj = optimizer.objective(xs_and_us)
    vio = jnp.linalg.norm(optimizer.constraints(xs_and_us))
    total_cost = obj + 1000*vio
    # print("total_cost", total_cost)
    return {'loss': total_cost, 'status': STATUS_OK}

  # define a search space
  space = {
      'maxiter': h.choice('maxiter', [i*1000 for i in range(1, 11)]),
      'eta_x_exp': h.uniform('eta_x_exp', -10, -4),
      'eta_v_exp': h.uniform('eta_v_exp', -6, -4),
      'atol_exp': h.uniform('atol_exp', -8, -4)
    }

  # pp.pprint(pyll.stochastic.sample(space))

  # minimize the objective over the space
  best = fmin(f, space, algo=tpe.suggest, max_evals=5)

  print(best)