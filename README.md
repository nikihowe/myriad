# optimal-control

## Directory structure
```
├── notebooks
    └── ...             # Unstructured one-off notebooks and scripts
├── paper
    └── ...             # ICLR 2021 paper draft
├── source
    ├── gin-configs     # Contains gin configuration files
        └── ...
    ├── LenhartSystems  # Dynamical systems presented in Lenhart book
        └── ...
    ├── config.py       # Configuration and data types
    ├── optimizers.py   # Trajectory optimization algorithms
    ├── systems.py      # Dynamical systems
    └── utils.py        # Helper methods
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

## Metrics
TODO: Set up wandb

## Tests
```bash
source env/bin/activate
python -m unittest discover -s tests
```

## References
- [Lenhart et Workman, *Optimal Control Applied to Biological Models*. Chapman and Hall/CRC, 2007.](https://www.taylorfrancis.com/books/9780429138058)
- [Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Pcrogramming*. SIAM Advances in Design and Control, 2010.](https://epubs.siam.org/doi/book/10.1137/1.9780898718577)
- [Kelly, *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*. SIAM Rev., 2017.](https://www.semanticscholar.org/paper/An-Introduction-to-Trajectory-Optimization%3A-How-to-Kelly/ba1f38d6bbbf7227cda93f3915bc3fa7fc37b58e)
## Citing this repo

```
@misc{optimal-control,
  author = {Howe, Nikolaus and Rajkumar, Nitarshan and Dufort-Labbé, Simon},
  title = {{what is the name (myriad?)}},
  note = {Available at: https://github.com/nikihowe/put_place_here},
  year = {2021}
}
```