"""This library implements in [JAX](https://github.com/google/jax) various systems and their associated OCP,
together with different trajectory_optimizers for their resolution"""
from .config import *
from .nlp_solvers import *
from .trajectory_optimizers import *
from .plotting import *
from .utils import *


# Exclude from documentation
__pdoc__ = {'trajectory_optimizers.IndirectMethodOptimizer.require_adj': False,
            'trajectory_optimizers.TrajectoryOptimizer.require_adj': False,
            'trajectory_optimizers.TrapezoidalCollocationOptimizer.require_adj': False,
            'trajectory_optimizers.HermiteSimpsonCollocationOptimizer.require_adj': False,
            'trajectory_optimizers.MultipleShootingOptimizer.require_adj': False,
            'trajectory_optimizers.IndirectMethodOptimizer.solve': False}
