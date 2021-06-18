"""This library implements in [JAX](https://github.com/google/jax) various systems and their associated OCP,
together with different optimizers for their resolution"""
from .config import *
from .nlp_solvers import *
from .optimizers import *
from .plotting import *
from .scripts import *
from .utils import *


__pdoc__ = {}
# Exclude from documentation
__pdoc__['optimizers.IndirectMethodOptimizer.require_adj'] = False
__pdoc__['optimizers.TrajectoryOptimizer.require_adj'] = False
__pdoc__['optimizers.TrapezoidalCollocationOptimizer.require_adj'] = False
__pdoc__['optimizers.HermiteSimpsonCollocationOptimizer.require_adj'] = False
__pdoc__['optimizers.MultipleShootingOptimizer.require_adj'] = False
__pdoc__['optimizers.IndirectMethodOptimizer.solve'] = False
__pdoc__["scripts"] = False