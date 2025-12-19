import numpy as np
import gymnasium as gym
from gonio_sim.envs.goniometer_env import GoniometerEnv, EnvConfig


class GoniometerRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 edge=8.6,
                 config="PARA",
                 phi="0/90",
                 nominal_edge=8.6,
                 max_steps=500):
        super().__init__()

        self.cfg = EnvConfig(edge=edge, config=config, phi=phi)
        self.nominal_edge = nominal_edge
        self.max_steps = max_steps

        self.env = GoniometerEnv(self.cfg)

        # Action space: pitch,yaw in {-1,0,+1}
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # Observation space: 7 variables
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        # Backlash tracking
        self.backlash_pitch_run = 0
        self.backlash_yaw_run = 0
        self.last_pitch_dir = 0
        self.last_yaw_dir = 0

        self._step_count = 0


    @staticmethod
    def _map_action_component(a):
        # {0,1,2} → {-1,0,+1}
        return a - 1


    def _action_to_deg(self, action):
        pitch_raw, yaw_raw = action
        dp = self._map_action_component(pitch_raw) * 1e-3
        dy = self._map_action_component(yaw_raw)  * 1e-3
        return dp, dy


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env = GoniometerEnv(self.cfg)
        self._step_count = 0

        # Reset backlash counters
        self.backlash_pitch_run = 0
        self.backlash_yaw_run = 0
        self.last_pitch_dir = 0
        self.last_yaw_dir = 0

        # Probe environment with a zero action
        phys = self.env.step(0.0, 0.0)

        peak      = float(phys[12])
        beam_delh = float(phys[6])
        beam_delv = float(phys[7])
        dose      = float(self.env.dose)

        sign_error = float(np.sign(peak - self.nominal_edge))

        obs = np.array([
            peak,
            dose,
            beam_delh,
            beam_delv,
            self.backlash_pitch_run,
            self.backlash_yaw_run,
            sign_error
        ], dtype=np.float32)

        return obs, {}


    def step(self, action):

        self._step_count += 1
        dp, dy = self._action_to_deg(action)

        # Direction signs for backlash tracking
        pitch_dir = int(np.sign(dp))
        yaw_dir   = int(np.sign(dy))

        # Update backlash counters
        if pitch_dir != 0:
            if pitch_dir == self.last_pitch_dir:
                self.backlash_pitch_run += 1
            else:
                self.backlash_pitch_run = 1
            self.last_pitch_dir = pitch_dir

        if yaw_dir != 0:
            if yaw_dir == self.last_yaw_dir:
                self.backlash_yaw_run += 1
            else:
                self.backlash_yaw_run = 1
            self.last_yaw_dir = yaw_dir

        # Physics step
        phys = self.env.step(dp, dy)

        peak      = float(phys[12])
        beam_delh = float(phys[6])
        beam_delv = float(phys[7])
        dose      = float(self.env.dose)
        sign_error = float(np.sign(peak - self.nominal_edge))

        # Build observation
        obs = np.array([
            peak,
            dose,
            beam_delh,
            beam_delv,
            self.backlash_pitch_run,
            self.backlash_yaw_run,
            sign_error
        ], dtype=np.float32)

        # Distance-based reward
        reward = -abs(peak - self.nominal_edge)

        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "dp_deg": dp,
            "dy_deg": dy,
            "dose": dose,
            "peak": peak,
            "error": peak - self.nominal_edge,
            "beam_delh": beam_delh,
            "beam_delv": beam_delv
        }

        return obs, reward, terminated, truncated, info


    def close(self):
        pass
