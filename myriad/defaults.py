# (c) 2021 Nikolaus Howe
from myriad.systems import SystemType


learning_rates = {
  SystemType.PENDULUM: {
    'eta_x': 1e-1,
    'eta_v': 1e-3
  },
  SystemType.CANCERTREATMENT: {  # works for single shooting, 50 controls
    'eta_x': 1e-1,
    'eta_v': 1e-3
  },
  SystemType.CARTPOLE: {
    'eta_x': 1e-2,
    'eta_v': 1e-4
  }
}

param_guesses = {
  SystemType.BACTERIA: {
    'r': 0.8,  # 1.
    'A': 1.2,  # 1.
    'B': 2.    # 1.
  },
  SystemType.BEARPOPULATIONS: {
    'r': .2,    # .1 (true values)
    'K': .6,    # .75
    'm_f': .3,  # .5
    'm_p': .6   # .5
  },
  SystemType.BIOREACTOR: {
    'D': 0.8,  # 1.
    'G': 1.2   # 1.
  },
  SystemType.PENDULUM: {
    'g': 15.,
    'm': 3.,
    'length': 0.5
  },
  SystemType.CARTPOLE: {
    'g': 10.,      # 9.81
    'm1': 1.5,     # 1.0
    'm2': 0.2,     # 0.3
    'length': 0.6  # 0.5
  },
  SystemType.CANCERTREATMENT: {
    'r': 0.1,  # 0.3
    # 'a': 0.1,  # NOTE: a is entirely for the cost, so we're not learning it for now
    'delta': 0.8  # 0.45
  },
  SystemType.GLUCOSE: {
    'a': 0.5,  # 1.
    'b': 0.4,  # 1.
    'c': 0.6   # 1.
  },
  SystemType.HIVTREATMENT: {
    'k': .000044,  # 0.000024
    'm_1': .01,    # 0.02
    'm_2': .9,     # 0.5
    'm_3': 3.4,    # 4.4
    'N': 250.,     # 300.
    'r': 0.02,     # 0.03
    's': 11.,      # 10.
    'T_max': 1400.  # 1500.
  },
  SystemType.MOULDFUNGICIDE: {
    'r': 0.1,  # 0.3
    'M': 8.   # 10.
  },
  SystemType.MOUNTAINCAR: {
    'power': 0.001,   # 0.0015
    'gravity': 0.005  # 0.0025
  },
  SystemType.PREDATORPREY: {
    'd_1': 0.15,  # 0.1
    'd_2': 0.07  # 0.1
  },
  SystemType.TIMBERHARVEST: {
    'k': 0.7  # 1.     # .4426
  },
  SystemType.TUMOUR: {
    'xi': 0.06,  # 0.084
    'b': 4.5,    # 5.85
    'd': 0.01,   # 0.00873
    'G': 0.2,    # 0.15
    'mu': 0.01   # 0.02
  },
  SystemType.VANDERPOL: {
    'a': 0.5  # 1.
  }
}
