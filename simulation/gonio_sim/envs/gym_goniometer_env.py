import gymnasium as gym
import numpy as np
from .goniometer_env import GoniometerEnv, EnvConfig


class GymGoniometerEnv(gym.Env):
metadata = {"render.modes": []}
def __init__(self):
self.env = GoniometerEnv(EnvConfig())
self.action_space = gym.spaces.MultiDiscrete([3, 3]) # pitch,yaw in {-1,0,+1}
self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
self.target = self.env.cfg.edge
def reset(self, *, seed=None, options=None):
self.env.reset(); obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
return obs, {}
def step(self, action):
dp = (action[0]-1) * 0.01 # 10 mdeg
dy = (action[1]-1) * 0.01
E = self.env.step(dp, dy)
err = self.target - E
reward = -abs(err)
obs = np.array([E, dp, dy], dtype=np.float32)
terminated = False
truncated = False
return obs, reward, terminated, truncated, {"error": err}
