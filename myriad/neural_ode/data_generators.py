# # (c) 2021 Nikolaus Howe
# from __future__ import annotations  # for nicer typing
#
# import typing
#
# if typing.TYPE_CHECKING:
#   pass
# import jax
# import jax.numpy as jnp
# import numpy as np
# import time
#
# from typing import Optional
#
# from myriad.config import Config, HParams, SamplingApproach
# from myriad.custom_types import Controls, Dataset
# from myriad.utils import integrate_time_independent_in_parallel, smooth
#
#
# def generate_dataset(hp: HParams, cfg: Config,
#                      given_us: Optional[Controls] = None) -> Dataset:
#   system = hp.system()
#   hp.key, subkey = jax.random.split(hp.key)
#
#   # Generate |total dataset size| control trajectories
#   total_size = hp.train_size + hp.val_size + hp.test_size
#
#   # TODO: fix what happens in case of infinite bounds
#   u_lower = system.bounds[hp.state_size:, 0]
#   u_upper = system.bounds[hp.state_size:, 1]
#   x_lower = system.bounds[:hp.state_size, 0]
#   x_upper = system.bounds[:hp.state_size, 1]
#   if jnp.isinf(u_lower).any() or jnp.isinf(u_upper).any():
#     raise Exception("infinite control bounds, aborting")
#   if jnp.isinf(x_lower).any() or jnp.isinf(x_upper).any():
#     raise Exception("infinite state bounds, aborting")
#
#   spread = (u_upper - u_lower) * hp.sample_spread
#
#   ########################
#   # RANDOM WALK CONTROLS #
#   ########################
#   if hp.sampling_approach == SamplingApproach.RANDOM_WALK:
#     # Make all the first states
#     all_start_us = np.random.uniform(u_lower, u_upper, (total_size, 1, hp.control_size))
#     all_us = all_start_us
#
#     for i in range(hp.num_steps):
#       next_us = np.random.normal(0, spread, (total_size, 1, hp.control_size))
#       rightmost_us = all_us[:, -1:, :]
#       together = np.clip(next_us + rightmost_us, u_lower, u_upper)
#       all_us = np.concatenate((all_us, together), axis=1)
#
#   # elif hp.sampling_approach == SamplingApproach.RANDOM_GRID:
#   #   single_ascending_controls = np.linspace(u_lower, u_upper, hp.num_steps + 1)
#   #   parallel_ascending_controls = single_ascending_controls[np.newaxis].repeat(total_size)
#   #   assert parallel_ascending_controls.shape == ()
#   # NOTE: we could also generate data by exhaustively considering every combination
#   #       of state-control pair up to some discretization. This might just solve
#   #       the problem. Unfortunately, curse of dimensionality is real.
#   # IDEA: let's try doing this on the CANCERTREATMENT domain, and see whether
#   #       this is enough to help neural ODE figure out what is going on
#   #       at the very start of planning
#
#   ###########################
#   # UNIFORM RANDOM CONTROLS #
#   ###########################
#   elif hp.sampling_approach == SamplingApproach.UNIFORM:
#     all_us = jax.random.uniform(subkey, (total_size, hp.num_steps + 1, hp.control_size),
#                                 minval=u_lower, maxval=u_upper) * 0.75  # TODO
#   # TODO: make sure having added control size everywhere didn't break things
#   #########################
#   # AROUND GIVEN CONTROLS #
#   #########################
#   elif hp.sampling_approach == SamplingApproach.TRUE_OPTIMAL or hp.sampling_approach == SamplingApproach.CURRENT_OPTIMAL:
#     if given_us is None:
#       print("Since you didn't provide any controls, we'll use a uniform random guess")
#       all_us = jax.random.uniform(subkey, (total_size, hp.num_steps + 1, hp.control_size),
#                                   minval=u_lower, maxval=u_upper) * 0.75  # TODO
#       # raise Exception("If sampling around a control trajectory, need to provide that trajectory.")
#
#     else:
#       noise = jax.random.normal(key=subkey, shape=(total_size, hp.num_steps + 1, hp.control_size)) \
#               * (u_upper - u_lower) * hp.sample_spread
#       all_us = jnp.clip(given_us[jnp.newaxis].repeat(total_size, axis=0).squeeze() + noise.squeeze(), a_min=u_lower,
#                         a_max=u_upper)
#
#   else:
#     raise Exception("Unknown sampling approach, please choose among", SamplingApproach.__dict__['_member_names_'])
#
#   print("initial controls shape", all_us.shape)
#
#   # Smooth the controls if so desired
#   if hp.to_smooth:
#     start = time.time()
#     all_us = smooth(all_us, 2)
#     end = time.time()
#     print(f"smoothing took {end - start}s")
#
#   # TODO: I really dislike having to have this line below. Is there no way to remove it?
#   # Make the controls guess smaller so our dynamics don't explode
#   # all_us *= 0.1
#
#   # Generate the start states
#   start_states = system.x_0[jnp.newaxis].repeat(total_size, axis=0)
#
#   # Generate the states from applying the chosen controls
#   if hp.start_spread > 0.:
#     hp.key, subkey = jax.random.split(hp.key)
#     start_states += jax.random.normal(subkey,
#                                       shape=start_states.shape) * hp.start_spread  # TODO: explore different spreads
#     start_states = jnp.clip(start_states, a_min=x_lower, a_max=x_upper)
#
#   # Generate the corresponding state trajectories
#   _, all_xs = integrate_time_independent_in_parallel(system.dynamics, start_states,
#                                                      all_us, hp.stepsize, hp.num_steps,
#                                                      hp.integration_method)
#
#   # Noise up the state observations
#   hp.key, subkey = jax.random.split(hp.key)
#   all_xs += jax.random.normal(subkey, shape=all_xs.shape) * (x_upper - x_lower) * hp.noise_level
#   all_xs = jnp.clip(all_xs, a_min=x_lower, a_max=x_upper)
#
#   # Stack the states and controls together
#   xs_and_us = jnp.concatenate((all_xs, all_us), axis=2)
#
#   if cfg.verbose:
#     print("Generating training control trajectories between bounds:")
#     print("  u lower", u_lower)
#     print("  u upper", u_upper)
#     print("of shapes:")
#     print("  xs shape", all_xs.shape)
#     print("  us shape", all_us.shape)
#     print("  together", xs_and_us.shape)
#
#   assert np.isfinite(xs_and_us).all()
#   return xs_and_us
#
#
# def yield_minibatches(hp: HParams, total_size: int, dataset: Dataset) -> iter:
#   assert total_size <= dataset.shape[0]
#
#   tmp_dataset = np.random.permutation(dataset)
#   num_minibatches = total_size // hp.minibatch_size + (1 if total_size % hp.minibatch_size > 0 else 0)
#
#   for i in range(num_minibatches):
#     n = np.minimum((i + 1) * hp.minibatch_size, total_size) - i * hp.minibatch_size
#     yield tmp_dataset[i * hp.minibatch_size: i * hp.minibatch_size + n]
#
#
# def sample_x_init(hp: HParams, n_batch: int = 1) -> np.ndarray:
#   s = hp.system()
#   res = np.random.uniform(s.bounds[:, 0], s.bounds[:, 1], (n_batch, hp.state_size + hp.control_size))
#   res = res[:, :hp.state_size]
#   assert np.isfinite(res).all()
#   return res
#
#
# if __name__ == "__main__":
#   hp = HParams()
#   cfg = Config()
#   dset = generate_dataset(hp, cfg)
#   # dset = np.random.rand(100, 5)
#   # hp = HParams()
#   # for e in yield_minibatches(hp, 91, dset):
#   #   print(e.shape)
#   # pass
#   # print(SamplingApproach.__dict__['_member_names_'])
#   # hp = HParams()
#   # n_batch = 10
#   # res = sample_x_init(hp, n_batch)
#   # print(res.shape)
#   #
#   # s = hp.system()
#   # lower = s.bounds[:, 0]
#   # upper = s.bounds[:, 1]
#   # res2 = np.random.uniform(s.bounds[:, 0],
#   #                          s.bounds[:, 1],
#   #                          (n_batch,
#   #                           hp.state_size + hp.control_size))  # keeping as is, though doesn't match our cartpole limits
#   # res2 = res2[:, :hp.state_size]
#   # print(res2.shape)
#
# # TODO: make a data generator, but with the optimal trajectories instead of random controls
# # def populate_data(hp: HParams, cfg: Config, system_params,
# #                   n_train, n_val, n_test, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #   np.random.seed(seed)
# #   n_data = n_train + n_val + n_test
# #   x_init = sample_x_init(hp=hp, n_batch=n_data)
# #
# #   system_params = {"x_0": x_init, **system_params}
# #
# #   system = hp.system(system_params)
# #   optimizer = get_optimizer(hp, cfg, system)
# #   solution = optimizer.solve()
# #
# #   x = solution['x']
# #   u = solution['u']
# #   if hp.order == IntegrationOrder.QUADRATIC and hp.optimizer == OptimizerType.COLLOCATION:
# #     x_mid = solution['x_mid']
# #     u_mid = solution['u_mid']
# #   if optimizer.require_adj:
# #     adj = solution['adj']
# #
# #   num_steps = hp.intervals * hp.controls_per_interval
# #   stepsize = system.T / num_steps
# #
# #   print("the shapes of x and u are", x.shape, u.shape)
# #
# #   #########
# #
# #   tau = np.cat((x, u), dim=2).transpose(0, 1)
# #   print("tau is", tau.shape)
# #   print("now splitting into train, val, and test")
# #
# #   train_data = tau[:n_train]
# #   val_data = tau[n_train:n_train + n_val]
# #   test_data = tau[n_train + n_val:]
# #
# #   return train_data, val_data, test_data
