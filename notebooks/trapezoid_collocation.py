# -*- coding: utf-8 -*-
"""Trapezoid Collocation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10_uYHRKoUqlcFnN4xCEwLq3wlxtVx0gR
"""

import numpy as onp
from scipy.optimize import minimize

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree # for putting in scipy minimize form

from jax.config import config
config.update("jax_enable_x64", True)

# Makes a cartpole function, given parameters
# mc:      mass of cart
# mp:      mass of pole
# length:  length of pole
# gravity: g
def make_cartpole(mc=2., mp=0.5, length=0.5, gravity=9.81):
    # Given state and control, returns velocities and accelerations
    def f(state, control):
        _, theta, x_vel, theta_vel = state
        # Page 868 of tutorial
        x_accel = (
            1/(mc + mp*(np.sin(theta)**2))
            * (
                control + mp*np.sin(theta)
                * (length*(theta_vel**2) + gravity*np.cos(theta))
            )
        )
        # Page 869 of tutorial
        theta_accel = (
            1/(length*(mc + mp*(np.sin(theta)**2)))
            * (-control*np.cos(theta) - mp*length*(theta_vel**2)*np.cos(theta)*np.sin(theta) - (mc + mp)*gravity*np.sin(theta))
        )
        return np.asarray([x_vel, theta_vel, x_accel, theta_accel])
    return f
    # f : (state, control) -> new state

def make_trapezoid_nlp(horizon, nintervals, unravel):

    nvars = nintervals + 1

    # Horizon sets how long each interval lasts
    interval_duration = horizon / nintervals
    # print("interval duration", interval_duration)

    f = make_cartpole()

    # Why do we del x?
    def cost(x, u):
        del x # good practice when you have a function that doesn't take an argument
        return u**2

    # Trapezoid equality constraints
    def trapezoid_defect(state, next_state, control, next_control):
        return (next_state - state) - (interval_duration/2)*(f(state, control) + f(next_state, next_control))

    # This is the "J" from the tutorial (6.5)
    def trapezoid_cost(state, next_state, control, next_control):
        return (interval_duration/2)*(cost(state, control) + cost(next_state, next_control))

    # Vectorizes the functions
    batched_cost = jax.vmap(trapezoid_cost)
    batched_defects = jax.vmap(trapezoid_defect)

    def objective(flat_variables):
        states, controls = unravel(flat_variables)
        # states:   (21,4) shaped array
        # controls: (21,)  shaped array

        # states[:-1]   is all but last row
        # states[1:]   is all but first row
        # controls[:-1] is all but last element
        # controls[1:] is all but first element

        # batched cost is vmapped, so this will take state = 0,   next_state = 1
                                                  #  state = 1,   next_state = 2
                                                  #  ...
                                                  #  state = T-1, next_state = T
        # and same for controls
        return np.sum(batched_cost(states[:-1], states[1:], controls[:-1], controls[1:]))

    def equality_constraints(flat_variables):
        states, controls = unravel(flat_variables)
        return np.ravel(batched_defects(states[:-1], states[1:], controls[:-1], controls[1:]))

    dist = 0.8
    umax = 100
    state_bounds = onp.empty((nvars, 8))
    # list of tuples, where each variable is a single scalar
    # it's a grid, because each row is the state vector at that time step
    # horizontal axis is 2x the number of variables in the state
    state_bounds[:, 0] = -2*dist # sets first  column to -2*dist
    state_bounds[:, 1] = 2*dist  # sets second column to +2*dist

    state_bounds[:, 2] = -2*onp.pi # then -2pi
    state_bounds[:, 3] = 2*onp.pi  # and  +2pi

    state_bounds[:, 4] = -onp.inf # velocity constraint
    state_bounds[:, 5] = onp.inf

    state_bounds[:, 6] = -onp.inf # angular velocity constraint
    state_bounds[:, 7] = onp.inf

    # Starting state
    state_bounds[0, :] = 0

    # Ending state
    state_bounds[-1, :] = 0
    state_bounds[-1, 0] = dist
    state_bounds[-1, 1] = dist
    state_bounds[-1, 2] = np.pi
    state_bounds[-1, 3] = np.pi

    # print("setting up state bounds")
    # np.set_printoptions(precision=3)
    # print(state_bounds.shape)
    # print(state_bounds)

    control_bounds = onp.empty((nvars, 2))
    control_bounds[:] = [-umax, umax]

    # print("setting up control bounds")
    # print(control_bounds.shape)
    # print(control_bounds)

    # state and control bounds are returned as a tuple of (state bounds, control bounds)
    # where state bounds is a list of 2-element upper-lower bound pairs
    # (take original 2D state bound array and reshape into rows of 2)
    return objective, equality_constraints, np.vstack(
        (np.reshape(state_bounds, (-1, 2)),
         control_bounds))

# Just a simple linear interpolation function
def control_interpolation(controls, interval_duration):
    def u(t):
        # Find which interval we're in (which two collocation points we're between
        kstart, kend = int(np.floor(t/interval_duration)), int(np.ceil(t/interval_duration))

        # If we're right on a collocation point, return the control value there
        if kstart == kend:
            return controls[kstart]

        # Starting time
        tstart = interval_duration * kstart

        # Equation 3.7, page 859
        scale = (t - tstart) / interval_duration
        return controls[kstart] + scale * (controls[kend] - controls[kstart])
    return u

# n: Interpolate the state
def state_interpolation(states, controls, interval_duration):
    def x(t):
        # We need the system dynamics to interpolate :)
        # TODO n: is there a nicer way to do this?
        f = make_cartpole()

        # Find which interval we're in (which two collocation points we're between
        kstart, kend = int(np.floor(t/interval_duration)), int(np.ceil(t/interval_duration))

        # If we're right on a collocation point, return the control value there
        if kstart == kend:
            return states[kstart]

        # Starting time
        tstart = interval_duration * kstart

        # Equation 3.10, page 859
        scale = (t - tstart) * (t - tstart) / (2 * interval_duration)
        return (states[kstart]
                + f(states[kstart], controls[kstart]) * (t - tstart)
                + scale * (f(states[kend], controls[kend]) - f(states[kstart], controls[kstart])))
    return x

import matplotlib.pyplot as plt

horizon = 2
intervals = 20

nvars = intervals + 1

dist = 0.8
# gives [0., 0.05, 0.1, ..., 0.9, 0.95, 1.]
linear_interpolation = np.arange(nvars)/(nvars-1)

# gives 21 rows of [0.8, pi, 0., 0.]
initial_states = np.tile(np.array([dist, np.pi, 0, 0]), (nvars, 1))

# multiply the above two, with broadcasting, to give
# (there are 20 timesteps, so 21 entries)
# [[0.   * 0.8, 0.   * pi, 0.   * 0., 0.   * 0.]
#  [0.05 * 0.8, 0.05 * pi, 0.05 * 0., 0.05 * 0.]
#  ...
#  [0.95 * 0.8, 0.95 * pi, 0.95 * 0., 0.95 * 0.]]
#  [1.   * 0.8, 1.   * pi, 1.   * 0., 1.   * 0.]]
# it's a linear interpolation in state between where we start and where we end
initial_states = linear_interpolation[:, np.newaxis] * initial_states

# a tuple, with (initial states, [0]*21)
initial_variables = (initial_states, np.zeros(nvars))

# flattens the initial variables into a big list
flat_initial_variables, unravel = ravel_pytree(initial_variables)

# objective takes flatten(state, control) variables, as does equality_constraints
# bounds takes the concatenated state and control bounds and packages them
# in groups of two [lower, upper], all in a long list. So the pairs of (parts of) state bounds all appear first,
# and the pairs of control bounds appears later in the long list
objective, equality_constraints, bounds = make_trapezoid_nlp(horizon, intervals, unravel)
# "horizon"   is the total time that you spend moving (here, 1)
# "intervals" is the number of intervals you divide the time into
# "unravel"   tells jax how to turn the variables into a form recognized by the NLP solver

# print("objective", objective)
# print("equality constraints", equality_constraints)
# print("bounds", bounds)

constraints = ({'type': 'eq',
                'fun': jax.jit(equality_constraints),
                'jac': jax.jit(jax.jacrev(equality_constraints))
                })

options = {'maxiter': 5000, 'ftol': 1e-6}

solution = minimize(fun=jax.jit(objective),
                    x0=flat_initial_variables,
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds,
                    jac=jax.jit(jax.grad(objective)),
                    options=options)

# print("solution", solution)
opt_states, opt_controls = unravel(solution.x)
# print("opt_states", opt_states[-1, :])

# Start the plot
fig, axs = plt.subplots(2, figsize=(8, 6))
fig.suptitle("Trapezoid Collocation Method")

# ax = fig.add_subplot(1, 1, 1)
time_axis = [(horizon/intervals) * k for k in range(nvars)]

# Plot the collocation points
major_ticks = np.arange(-1, 2.1, 0.2)
minor_ticks = np.arange(-1, 2.1, 0.05)
y_ticks = np.arange(-2, 4, 0.2)
minor_y_ticks = np.arange(-2, 4, 0.1)

axs[0].set_xticks(major_ticks)
axs[0].set_xticks(minor_ticks, minor=True)
axs[0].set_yticks(y_ticks)
axs[0].set_yticks(minor_y_ticks, minor=True)

major_ticks2 = np.arange(-30, 30, 10)
minor_ticks2 = np.arange(-30, 30, 5)

axs[1].set_xticks(major_ticks)
axs[1].set_xticks(minor_ticks, minor=True)
axs[1].set_yticks(major_ticks2)
axs[1].set_yticks(minor_ticks2, minor=True)

# ax.plot(time_axis, opt_states[:, 1], 'o', color="green", label="angle") # Angle

# Generate denser points for interpolation between collocation points
time_dense = np.linspace(0, horizon, 101)

# Get interpolation functions for control (linear) and state (quadratic)
linear_control = control_interpolation(opt_controls, horizon/intervals)
quadratic_state = state_interpolation(opt_states, opt_controls, horizon/intervals)

# Get the interpolated control and state values
controls = np.array([linear_control(t) for t in time_dense])
states = np.array([quadratic_state(t) for t in time_dense])

# Plot the state
axs[0].plot(time_axis, opt_states[:, 0], marker='o', linestyle='', fillstyle="none", color="blue", label="position (collocation)")  # Position
axs[0].plot(time_dense, states[:,0], marker='.', linestyle='', markersize=3, color="blue", alpha=0.5, label="position (interpolated)")  # Position
# ax.plot(time_dense, states[:,1], marker='+', color="red") # Angle
axs[1].plot(time_axis, opt_controls, marker='o', linestyle='', fillstyle="none", color="red", label='control (collocation)')       # Control
axs[1].plot(time_dense, controls, marker='.', linestyle='', markersize=3, alpha=0.5, color="red", label="control (interpolated)")      # Control

# Add legend, grid, and show plot
for ax in axs:
    ax.legend(loc=4)

axs[1].set(xlabel='time', ylabel='')
axs[1].set_ylim([-30, 22])

axs[0].grid(which="both")
axs[0].grid(which='minor', alpha=0.2)
axs[0].grid(which='major', alpha=0.5)
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)
plt.show()
