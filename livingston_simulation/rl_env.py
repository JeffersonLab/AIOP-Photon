import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from livingston_sim import CoherentBremsstrahlungSimulator


class CoherentGoniometerEnv(gym.Env):

    metadata = {"render_modes": []}

    # Orientation mapping (authoritative)
    # TODO: previous vibe-session ordering (kept for verification):
    # ORIENTATIONS = ["PARA 0/90", "PERP 0/90", "PARA 45/135", "PERP 45/135"]
    ORIENTATIONS = [
        # "AMORPHOUS",
        "PERP 0/90",
        "PARA 0/90",
        "PERP 45/135",
        "PARA 45/135",
    ]

    def __init__(
        self,
        beam_energy_E0=11600.0,     # MeV
        coherent_edge_Ei=8600.0,    # MeV
        base_peak_position=8600.0, # MeV
        dose_slope=0.05,
        orientation_index=0,
        run_period="2020",
        pitch_step_deg=1e-3,
        yaw_step_deg=1e-3,
        dose_per_step=0.1,
        max_steps=500,

        # use_streamlined_peaks=True,
        # nudge_energy_size_pitch=10.0,
        # nudge_energy_size_yaw=10.0,
        # latency_setpoint_to_readback=8,
    ):
        super().__init__()

        self.beam_energy_E0 = beam_energy_E0
        self.coherent_edge_Ei = coherent_edge_Ei
        self.pitch_step_deg = pitch_step_deg
        self.yaw_step_deg = yaw_step_deg
        self.dose_per_step = dose_per_step
        self.max_steps = max_steps
        
        #### for normalization
        self.MAX_ENERGY = 12000.0  # MeV
        self.MAX_DOSE = 100.0 
        
        # self.use_streamlined_peaks = use_streamlined_peaks
        # self.nudge_energy_size_pitch = nudge_energy_size_pitch
        # self.nudge_energy_size_yaw = nudge_energy_size_yaw
        # self.latency_setpoint_to_readback = latency_setpoint_to_readback

        self.run_period = run_period

        # Orientation handling
        if not (0 <= orientation_index < len(self.ORIENTATIONS)):
            raise ValueError("orientation_index must be in {0,1,2,3,4}")

        self.orientation_index = orientation_index
        self.orientation_label = self.ORIENTATIONS[orientation_index]

        if self.orientation_label == "AMORPHOUS":
            self.coherent_edge_Ei = 0.0
            base_peak_position = 0.0

        self.pitch_action_history = deque(maxlen=5)  # store only ±1
        self.yaw_action_history   = deque(maxlen=5)

        
        # Simulator
        self.sim = CoherentBremsstrahlungSimulator(
            base_peak_position=base_peak_position,
            dose_slope=dose_slope,
            beam_energy_E0=self.beam_energy_E0,
            coherent_edge_Ei=self.coherent_edge_Ei,
            orientation=self.orientation_label,
            run_period=self.run_period,

            # use_streamlined_energy=self.use_streamlined_peaks,
            # nudge_energy_size_pitch=self.nudge_energy_size_pitch,
            # nudge_energy_size_yaw=self.nudge_energy_size_yaw,
            # latency_setpoint_to_readback=self.latency_setpoint_to_readback,
        )

        # Action space: pitch, yaw ∈ {-1,0,+1}
        self.action_space = spaces.MultiDiscrete([3, 3])

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,        # beam energy
                0.0,        # coherent edge
                # -np.inf,    # true pitch
                # -np.inf,    # true yaw
                0.0,        # dose
                0.0,        # orientation index
                -1.0,       # sign error
                -1.0, -1.0, -1.0, -1.0, -1.0,  # last 5 pitch actions
                -1.0, -1.0, -1.0, -1.0, -1.0   # last 5 yaw actions
            ], dtype=np.float32),
            high=np.array([
                1.0, # np.inf,
                1.0, # np.inf,
                # 1.0, # np.inf,
                # 1.0, # np.inf,
                1.0, # np.inf,
                3.0,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0
            ], dtype=np.float32),
            dtype=np.float32
        )


        self._step_count = 0

    def _pad_history(self, history, length=5):
        padded = [0.0] * (length - len(history)) + list(history)
        return padded
        

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

        pitch_hist = self._pad_history(self.pitch_action_history)
        yaw_hist   = self._pad_history(self.yaw_action_history)

        norm_beam_E = self.beam_energy_E0 / self.MAX_ENERGY
        norm_coh_E  = self.coherent_edge_Ei / self.MAX_ENERGY
        norm_dose   = self.sim.dose.dose / self.MAX_DOSE
        return np.array([
            norm_beam_E,
            norm_coh_E,
            # self.sim.goni.return_diamond_pitch(),
            # self.sim.goni.return_diamond_yaw(),
            norm_dose,
            self.orientation_index,
            sign_error,
            *pitch_hist,
            *yaw_hist
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Gym API
    # ----------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim = CoherentBremsstrahlungSimulator(
            base_peak_position=self.coherent_edge_Ei,
            dose_slope=self.sim.peak_tracker.dose_slope,
            beam_energy_E0=self.beam_energy_E0,
            coherent_edge_Ei=self.coherent_edge_Ei,
            orientation=self.orientation_label,
            run_period=self.run_period,

            # use_streamlined_energy=self.use_streamlined_peaks,
            # nudge_energy_size_pitch=self.nudge_energy_size_pitch,
            # nudge_energy_size_yaw=self.nudge_energy_size_yaw,
            # latency_setpoint_to_readback=self.latency_setpoint_to_readback,
        )

        self._step_count = 0

        delta_c, peak = self.sim.step(
            dpitch_deg=0.0,
            dyaw_deg=0.0,
            delta_dose=0.0
        )

        self.pitch_action_history.clear()
        self.yaw_action_history.clear()

        return self._get_obs(peak), {}


    def step(self, action):
        self._step_count += 1
        
        pitch_a, yaw_a = action

        pitch_dir = self._map_action(pitch_a)
        yaw_dir   = self._map_action(yaw_a)
        
        if pitch_dir != 0:
            self.pitch_action_history.append(pitch_dir)
            
        if yaw_dir != 0:
            self.yaw_action_history.append(yaw_dir)
        
        dp = self._map_action(pitch_a) * self.pitch_step_deg
        dy = self._map_action(yaw_a)   * self.yaw_step_deg


        prev_peak = self.sim.peak_tracker.current_peak_position

        delta_c, peak = self.sim.step(
            dpitch_deg=dp,
            dyaw_deg=dy,
            delta_dose=self.dose_per_step
        )

        
        error = peak - self.coherent_edge_Ei
        relative_error = abs(peak - self.coherent_edge_Ei) / (self.coherent_edge_Ei + 1e-8)
        
        #### Reward: minimize distance from coherent edge
        # reward = -abs(error)
        reward = self.calculate_reward(relative_error)

        action_penalty = (pitch_dir**2 + yaw_dir**2)**0.5 * 0.01  # small penalty for taking an action
        reward -= action_penalty

        # Termination
        if np.absolute(error) / self.coherent_edge_Ei < 0.001:
            terminated = True
        else:
            terminated = False


        if self._step_count >= self.max_steps:
            terminated = True

        truncated = False

        info = {
            "peak": peak,
            "error": error,
            "delta_c": delta_c,
            "dose": self.sim.dose.dose,
            "pitch_deg": self.sim.goni.return_diamond_pitch(),
            "yaw_deg": self.sim.goni.return_diamond_yaw(),
            "orientation": self.orientation_label
        }

        return self._get_obs(peak), reward, terminated, truncated, info

    def calculate_reward(self, relative_error):
        a = 10  # Maximum reward
        b = 0.01  # Controls the steepness of the curve
        c = 0.001  # Controls the center point of the curve
        safe_exp = np.clip((relative_error - c) / b, -709, 709)  # Prevent exp overflow
        return a / (1 + np.exp(safe_exp))

    def close(self):
        pass
