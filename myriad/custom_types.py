# (c) Nikolaus Howe 2021
import jax.numpy as jnp

from typing import Callable, Mapping, Optional, Union


Batch = jnp.ndarray
Control = Union[float, jnp.ndarray]
Controls = jnp.ndarray
Cost = float
Dataset = jnp.ndarray
Defect = jnp.ndarray
DParams = Mapping[str, Union[float, jnp.ndarray]]
DState = Union[float, jnp.ndarray]
DStates = jnp.ndarray
Epoch = int
Params = Mapping[str, Union[float, jnp.ndarray]]
Solution = Mapping[str, Union[float, jnp.ndarray]]
State = Union[float, jnp.ndarray]
States = jnp.ndarray
Timestep = int

CostFun = Callable[[State, Control, Optional[Timestep]], Cost]
DynamicsFun = Callable[[State, Control, Optional[Timestep]], DState]
