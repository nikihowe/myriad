# optimal-control

## Directory structure
```
├── notebooks
    └── ...             # Unstructured one-off notebooks and scripts
├── source
    ├── config.py       # Configuration and data types
    ├── optimizers.py   # Trajectory optimization algorithms
    ├── systems.py      # Dynamical systems
    └── utils.py        # Helper methods
├── tests
    └── ...             # Automated tests
└── run.py              # Entry point for all experiments
```

## Environment setup
```bash
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Running experiments
```bash
source env/bin/activate
python run.py --system=CARTPOLE --optimizer=COLLOCATION
python run.py --system=CARTPOLE --optimizer=SHOOTING --intervals=50
python run.py --system=VANDERPOL --optimizer=SHOOTING --intervals=1 --controls_per_interval=50
python run.py --system=SEIR --optimizer=SHOOTING --slsqp_maxiter=500
# etc.
```

## Metrics
TODO: Set up wandb

## Tests
```bash
source env/bin/activate
python -m unittest discover -s tests
```
