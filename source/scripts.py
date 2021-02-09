# # --------------------------------------
# # Script for running the standard trajectory optimization
# #   put this in run.py
#   optimizer = get_optimizer(hp, cfg, system)
#   if optimizer.require_adj:
#     x, u, adj = optimizer.solve()
#   else:
#     x, u = optimizer.solve()
#
#   num_steps = hp.intervals*hp.controls_per_interval
#   stepsize = system.T / num_steps
#   _, opt_x = integrate(system.dynamics, system.x_0, u,
#                       stepsize, num_steps, None, hp.order)
#
#   if cfg.plot_results:
#     if optimizer.require_adj:
#       plot(system,
#            data={'x': x, 'u': u, 'adj': adj, 'other_x': opt_x},
#            labels={'x': ' (from solver)',
#                    'u': 'Controls from solver',
#                    'adj': 'Adjoint from solver',
#                    'other_x': ' (from integrating controls from solver)'})
#     else:
#       plot(system,
#            data={'x': x, 'u': u, 'other_x': opt_x},
#            labels={'x': ' (from solver)',
#                    'u': 'Controls from solver',
#                    'other_x': ' (from integrating controls from solver)'})
#
#   xs_and_us, unused_unravel = jax.flatten_util.ravel_pytree((x, u))
#   if hp.optimizer != OptimizerType.FBSM:
#     print("control cost", optimizer.objective(xs_and_us))
#     print('constraint_violations', jnp.linalg.norm(optimizer.constraints(xs_and_us)))
#   raise SystemExit
#
# # --------------------------------------------