# Myriad

Myriad is a real-world testbed that aims to bridge the gap between trajectory optimization and deep learning.
Myriad offers many real-world relevant,
continuous-time dynamical system environments, and several trajectory optimization algorithms.
These are all written
in [JAX](https://github.com/google/jax), and as such can be easily integrated into a deep learning workflow.
The environments and tools in Myriad can be used for trajectory optimization, system identification, 
imitation learning, and reinforcement learning.

It is our hope that Myriad will
serve as a stepping stone towards the increased development
of machine learning algorithms with the goal of addressing
impactful real-world challenges.

For an overview of the motivation and capabilities of Myriad, see our [arXiv preprint](https://arxiv.org/abs/2202.10600).

For implementation and usage details, see the Myriad [documentation](https://nikihowe.github.io/myriad/html/myriad/index.html).

## Environment setup

To run the examples and experiments in Myriad, you must first set up the environment.
This involves installing `ipopt` on your system, as well as installing the Myriad dependencies.
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

## Running the examples

Now that you have installed the necessary dependencies, you can run some examples using the tools in Myriad.
The easiest place to start is the Jupyter notebooks we provide.
To run these, you'll need to install Jupyter notebook as well as the Myriad package itself,
which will enable the notebook to see the packages it needs.

### Install Jupyter notebook and Myriad
We can install Jupyter notebooks and Myriad by running
```bash
pip install notebook
python setup.py build
python setup.py install
```

### Running the notebook

Then we can run the notebooks by navigating to the notebook directory and running launching a notebook
```bash
cd myriad/examples
jupyter notebook
```

## Performing experiments

To see what is going on "under the hood" in the various examples mentioned above, 
you can look, modify, and run the code in the `experiments` directory.

For example, to run trajectory optimization, uncomment line 54 of `run.py`,
and then run
```bash
python run.py
```

You can modify hyperparameters either by changing the defaults directly in the `config.py` file, 
or by passing them in as arguments to the `run.py` script.
For example, you can try the following:

```bash
python run.py --system=SIMPLECASE --optimizer=SHOOTING
python run.py --system=CARTPOLE --optimizer=COLLOCATION --intervals=100
python run.py --system=VANDERPOL --optimizer=SHOOTING --intervals=1 --controls_per_interval=50
python run.py --system=CANCERTREATMENT --optimizer=SHOOTING --max_iter=500
```

To run the other experiments, such as system identification (with parametric or Neural ODE model)
or end-to-end control-oriented imitation learning (with parametric or Neural ODE model), 
try uncommenting the corresponding line in `run.py` (60, 63, 69, 72), and looking at the 
corresponding code in the `experiments` directory.

## Directory structure
```
├── myriad
    ├── examples        # Various examples presented in Jupyter notebooks
        └── ...
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
    ├── study_scripts.py   # To study learned dynamics and effect of noise
    ├── useful_scripts.py  # Scripts for various tasks
    └── utils.py        # Simple helper functions
├── run.py              # Entry point for running code
├── setup.py            # To install Myriad
└── tests               # Tests
    └── ...
```

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
