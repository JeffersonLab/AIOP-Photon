import numpy as np
import gymnasium as gym
from gymnasium import spaces

from livingston_sim import CoherentBremsstrahlungSimulator


class CoherentGoniometerEnv(gym.Env):

    metadata = {"render_modes": []}

    # Orientation mapping (authoritative)
    ORIENTATIONS = [
        "PARA 0/90",
        "PERP 0/90",
        "PARA 45/135",
        "PERP 45/135",
    ]

    def __init__(
        self,
        beam_energy_E0=11600.0,     # MeV
        coherent_edge_Ei=8600.0,    # MeV
        base_peak_position=8600.0, # MeV
        dose_slope=0.05,
        orientation_index=0,
        pitch_step_deg=1e-3,
        yaw_step_deg=1e-3,
        dose_per_step=0.1,
        max_steps=500,
    ):
        super().__init__()

        self.beam_energy_E0 = beam_energy_E0
        self.coherent_edge_Ei = coherent_edge_Ei
        self.pitch_step_deg = pitch_step_deg
        self.yaw_step_deg = yaw_step_deg
        self.dose_per_step = dose_per_step
        self.max_steps = max_steps

        # Orientation handling
        if not (0 <= orientation_index < 4):
            raise ValueError("orientation_index must be in {0,1,2,3}")

        self.orientation_index = orientation_index
        self.orientation_label = self.ORIENTATIONS[orientation_index]

        # Simulator
        self.sim = CoherentBremsstrahlungSimulator(
            base_peak_position=base_peak_position,
            dose_slope=dose_slope,
            beam_energy_E0=self.beam_energy_E0,
            coherent_edge_Ei=self.coherent_edge_Ei,
            orientation=self.orientation_label,
        )

        # Action space: pitch, yaw ∈ {-1,0,+1}
        self.action_space = spaces.MultiDiscrete([3, 3])

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,        # beam energy
                0.0,        # coherent edge
                -np.inf,    # pitch
                -np.inf,    # yaw
                0.0,        # dose
                0.0,         # orientation index
                -1.0        #sign error
            ]),
            high=np.array([
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                3.0,
                1.0
            ]),
            dtype=np.float32
        )

        self._step_count = 0


    def _sign_error(self, peak):
        """
        +1 if peak > nominal coherent edge
        -1 if peak < nominal coherent edge
        """
        return 1.0 if peak > self.coherent_edge_Ei else -1.0
    
        
    # ----------------------------------------------------
    # Utilities
    # ----------------------------------------------------

    @staticmethod
    def _map_action(a):
        """Map {0,1,2} → {-1,0,+1}"""
        return int(a) - 1

    def _get_obs(self, peak):
        sign_error = self._sign_error(peak)
        return np.array([
            self.beam_energy_E0,
            self.coherent_edge_Ei,
            self.sim.state.pitch_deg,
            self.sim.state.yaw_deg,
            self.sim.dose.dose,
            self.orientation_index,
            sign_error
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Gym API
    # ----------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim = CoherentBremsstrahlungSimulator(
            base_peak_position=self.coherent_edge_Ei,
            dose_slope=self.sim.peak.dose_slope,
            beam_energy_E0=self.beam_energy_E0,
            coherent_edge_Ei=self.coherent_edge_Ei,
            orientation=self.orientation_label,
        )

        self._step_count = 0

        delta_c, peak = self.sim.step(
            dpitch_deg=0.0,
            dyaw_deg=0.0,
            delta_dose=0.0
        )

        return self._get_obs(peak), {}


    def step(self, action):
        self._step_count += 1

        pitch_a, yaw_a = action
        dp = self._map_action(pitch_a) * self.pitch_step_deg
        dy = self._map_action(yaw_a)   * self.yaw_step_deg

        delta_c, peak = self.sim.step(
            dpitch_deg=dp,
            dyaw_deg=dy,
            delta_dose=self.dose_per_step
        )

        # Reward: minimize distance from coherent edge
        error = peak - self.coherent_edge_Ei
        reward = -abs(error)

        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "peak": peak,
            "error": error,
            "delta_c": delta_c,
            "dose": self.sim.dose.dose,
            "pitch_deg": self.sim.state.pitch_deg,
            "yaw_deg": self.sim.state.yaw_deg,
            "orientation": self.orientation_label
        }

        return self._get_obs(peak), reward, terminated, truncated, info

    def close(self):
        pass
