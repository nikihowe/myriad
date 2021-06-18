# (c) Nikolaus Howe 2021
import jax.numpy as jnp

from typing import Union

Dataset = jnp.ndarray
Batch = jnp.ndarray
Control = float
Controls = jnp.ndarray
State = Union[float, jnp.ndarray]
DState = Union[float, jnp.ndarray]
States = jnp.ndarray
Cost = float
Epoch = int
Time = float
