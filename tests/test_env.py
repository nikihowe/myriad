# import unittest
#
# import jax
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import PPO
#
# from source.config import IntegrationOrder as IO
# from source.gym import FiniteHorizonControlEnv
# from source.systems import CartPole
#
#
# class EnvTest(unittest.TestCase):
#   def setUp(self):
#     jax.config.update("jax_enable_x64", True)
#     self.env = FiniteHorizonControlEnv(CartPole(), 100, IO.QUADRATIC)
#
#   def test_env(self):
#     check_env(self.env)
#
#   def test_env_ppo(self):
#     model = PPO('MlpPolicy', self.env, n_steps=100, n_epochs=20, batch_size=100, learning_rate=0.001)
#     model.learn(1000)
#
#     obs = self.env.reset()
#     while True:
#       action, _states = model.predict(obs)
#       obs, rewards, dones, info = self.env.step(action)
#       if dones == True:
#         break
#     self.env.render()
#
#
# if __name__=='__main__':
#   unittest.main()
