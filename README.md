# Myriad

Myriad is a collection of dynamical system environments in which to perform optimal control,
along with a collection of optimization techniques which can be applied to them in
a mix-and-match fashion.

The aim of this repo is promote and support the development of learning algorithms applicable to a wide
range of real-world problems.

For descriptions of the available environments, optimizers, and other
aspects of this repository, please see the [documentation](https://nikihowe.github.io/optimal-control/html/myriad/index.html).

## Directory structure
```
├── myriad
    ├── gin-configs     # Contains gin configuration files
        └── ...
    ├── systems         # Dynamical systems
        └── ...
    ├── config.py       # Configuration and data types
    ├── nlp_solvers     # Nonlinear program solvers
    ├── optimizers.py   # Trajectory optimization algorithms
    ├── plotting.py     # Code for plotting results
    ├── scripts.py      # Some useful scripts
    └── utils.py        # Helper functions
├── tests
    └── ...             # Automated tests
└── run.py              # Entry point for all experiments
```

## Environment setup
### IPOPT (Linux)
```bash
apt install coinor-libipopt-dev
```

### IPOPT (Mac OS)
```bash
brew install ipopt
brew install pkg-config
```

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Gin

```bash
pip install gin-config
```

## Running experiments
```bash
source env/bin/activate
python run.py --system=CARTPOLE --optimizer=COLLOCATION
python run.py --system=CARTPOLE --optimizer=SHOOTING --intervals=50
python run.py --system=VANDERPOL --optimizer=SHOOTING --intervals=1 --controls_per_interval=50
python run.py --system=SEIR --optimizer=SHOOTING --ipopt_max_iter=500
# etc.
```

### Parameters specification with gin-config
All Lenhart dynamic system can have their specific parameters default values modified via the `gin_bindings` command.
Multiple parameters can be specified by reusing the command. Example:
```bash
python run.py --system=GLUCOSE --optimizer=FBSM \
    --gin_bindings="Glucose.l=0.4" \
    --gin_bindings="Glucose.T=0.3" 
# etc.
```

## Tests
```bash
source env/bin/activate
python -m unittest discover -s tests
```
There will be more test documentation soon.

## References
- [Lenhart et Workman, *Optimal Control Applied to Biological Models*. Chapman and Hall/CRC, 2007.](https://www.taylorfrancis.com/books/9780429138058)
- [Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Pcrogramming*. SIAM Advances in Design and Control, 2010.](https://epubs.siam.org/doi/book/10.1137/1.9780898718577)
- [Kelly, *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*. SIAM Rev., 2017.](https://www.semanticscholar.org/paper/An-Introduction-to-Trajectory-Optimization%3A-How-to-Kelly/ba1f38d6bbbf7227cda93f3915bc3fa7fc37b58e)
## Citing this repo

```
@misc{optimal-control,
  author = {Howe, Nikolaus and Dufort-Labbé, Simon and Rajkumar, Nitarshan},
  title = {Myriad},
  note = {Available at: https://github.com/nikihowe/myriad},
  year = {2021}
}
```
