"""
This library implements in [JAX](https://github.com/google/jax) various real-world environments,
neural ODEs for system identification, and trajectory optimizers for solving the optimal control problem.
"""
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
