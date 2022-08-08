# Myriad

Myriad is a real-world testbed that aims to bridge the gap between
trajectory optimization and deep learning. Myriad offers many real-world relevant,
continuous space and time dynamical system environments for optimal control.
Myriad is written in JAX, and both environments and trajectory optimization
routines are fully differentiable. The tools in Myriad can be used
for trajectory optimization, system identification, imitation learning, and
reinforcement learning.

The aim of this repository is to offer trajectory
optimization tools to the machine learning community
in a way that can be seamlessly integrated in deep learning workflow.
Simultaneously, we hope that Myriad will
serve as a stepping stone towards the increased development
of machine learning algorithms with the goal of addressing
real-world problems.

For an overview of the motivation and capabilities of Myriad, see https://arxiv.org/abs/2202.10600.

For implementation and usage details, see the [documentation](https://nikihowe.github.io/optimal-control/html/myriad/index.html).

## Directory structure
```
├── myriad
    ├── experiments     # Various experiments (see Section 6 of arXiv paper)
        └── ...
    ├── gin-configs     # Default environment parameters
        └── ...
    ├── neural_ode      # Infrastructure for Neural ODE dynamics models
        └── ...
    ├── nlp_solvers     # Nonlinear Program Solvers
        └── ...
    ├── systems         # Environments
        └── ...
    ├── trajectory_optimizers     # Trajectory optimization routines
        └── ...
    ├── config.py       # Training and environment hyperparameters
    ├── custom-types    # Project types
    ├── defaults        # Learning rates and SysID parameter guesses
    ├── plotting.py     # Code for plotting results
    ├── study_scripts.py          # To study learned dynamics and effect of noise
    ├── useful_scripts.py         # Scripts for various tasks
    └── utils.py        # Simple helper functions
├── tests               # Tests
    └── ...
└── run.py              # Entry point for running code
```

## Environment setup

### Install IPOPT
#### Linux
```bash
apt install coinor-libipopt-dev
```

#### Mac OS
```bash
brew install pkg-config
brew install ipopt
```

### Create a venv and install requirements
```bash
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Running experiments

### Trajectory Optimization
Uncomment line 64 of `run.py`. Then:
```bash
source env/bin/activate
python run.py --system=CARTPOLE --optimizer=COLLOCATION
python run.py --system=CARTPOLE --optimizer=SHOOTING --intervals=50
python run.py --system=VANDERPOL --optimizer=SHOOTING --intervals=1 --controls_per_interval=50
python run.py --system=SEIR --optimizer=SHOOTING --ipopt_max_iter=500
# etc.
```

### Parameters specification with gin-config
```bash
python run.py --system=GLUCOSE --optimizer=FBSM \
    --gin_bindings="Glucose.l=0.4" \
    --gin_bindings="Glucose.T=0.3" 
# etc.
```

### Running learning experiments
Uncomment relevant line in `run.py` (70, 73, 79, 82). Then
do the same as above.

## References
- [Lenhart et Workman, *Optimal Control Applied to Biological Models*. Chapman and Hall/CRC, 2007.](https://www.taylorfrancis.com/books/9780429138058)
- [Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Pcrogramming*. SIAM Advances in Design and Control, 2010.](https://epubs.siam.org/doi/book/10.1137/1.9780898718577)
- [Kelly, *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*. SIAM Rev., 2017.](https://www.semanticscholar.org/paper/An-Introduction-to-Trajectory-Optimization%3A-How-to-Kelly/ba1f38d6bbbf7227cda93f3915bc3fa7fc37b58e)

## Citing Myriad
```
@article{howe2022myriad,
  title={Myriad: a real-world testbed to bridge trajectory optimization and deep learning},
  author={Howe, Nikolaus HR and Dufort-Labb{\'e}, Simon and Rajkumar, Nitarshan and Bacon, Pierre-Luc},
  journal={arXiv preprint arXiv:2202.10600},
  year={2022}
}
```
