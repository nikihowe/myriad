# optimal-control

## Directory structure
```
├── notebooks
    └── ...
├── source
    └── ...
├── tests
    └── ...
└── run.py    # Entry point for all experiments
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
python run.py --dynamics=CARTPOLE
```

## Metrics
TODO: Set up wandb

## Tests
```bash
source env/bin/activate
python -m unittest discover -s tests
```
