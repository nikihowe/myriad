import unittest

import jax
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
#from stable_baselines3.common.cmd_util import make_vec_env

from source.config import IntegrationOrder as IO
from source.gym import FiniteHorizonControlEnv
from source.systems import CartPole


class EnvTest(unittest.TestCase):
  def test_env(self):
    jax.config.update("jax_enable_x64", True)
    env = FiniteHorizonControlEnv(CartPole(), 100, IO.QUADRATIC)
    check_env(env)
  
  def test_env_ppo(self):
    jax.config.update("jax_enable_x64", True)
    env = FiniteHorizonControlEnv(CartPole(), 100, IO.QUADRATIC)
    #env = make_vec_env(FiniteHorizonControlEnv, 1, env_kwargs={'system': CartPole(), 'intervals': 10, 'integration_order': IO.QUADRATIC})
    model = PPO('MlpPolicy', env, verbose=2, n_steps=25, n_epochs=20, batch_size=100, learning_rate=0.001, clip_range=0.2, ent_coef=0.0, gamma=0.98)
    model.learn(10_000)


if __name__=='__main__':
  unittest.main()
