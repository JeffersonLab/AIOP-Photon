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
        target_edge_low=8400.0,     # MeV, for randomization
        target_edge_high=8800.0,    # MeV, for randomization
        dose_slope=0.05,
        orientation_index=0,
        run_period="2020",
        pitch_step_deg=2e-4,
        yaw_step_deg=2e-4,
        dose_per_step=0.1,
        max_steps=500,

        # use_streamlined_peaks=True,
        # nudge_energy_size_pitch=10.0,
        # nudge_energy_size_yaw=10.0,
        # latency_setpoint_to_readback=8,
    ):
        super().__init__()

        # target edge position
        coherent_edge_Ei = self.np_random.uniform(low=target_edge_low, high=target_edge_high)

        sign = self.np_random.choice([-1.0, 1.0])
        magnitude = self.np_random.uniform(low=10.0, high=50.0)
        peak_offset = sign * magnitude
        base_peak_position = coherent_edge_Ei + peak_offset

        self.beam_energy_E0 = beam_energy_E0
        self.target_edge_low = target_edge_low
        self.target_edge_high = target_edge_high
        self.coherent_edge_Ei = coherent_edge_Ei
        self.pitch_step_deg = pitch_step_deg
        self.yaw_step_deg = yaw_step_deg
        self.dose_per_step = dose_per_step
        self.max_steps = max_steps

        self.steps_stable_counter = 0
        
        #### for normalization
        self.MAX_ENERGY = 12000.0  # MeV
        self.MAX_DOSE = 500.0 
        
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

        self.action_history = deque(maxlen=5)

        
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

        # action space: a ∈ {-1,0,+1} just one action
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,        # beam energy
                0.0,        # coherent edge (target)
                0.0,        # current peak position
                0.0,        # relative error
                0.0,        # dose
                0.0,        # orientation index
                -1.0,       # sign error
                -1.0, -1.0, -1.0, -1.0, -1.0  # last 5 actions
            ], dtype=np.float32),
            high=np.array([
                1.0, # np.inf,
                1.0, # np.inf,
                1.0, 
                1.0,
                1.0, # np.inf,
                3.0,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0
            ], dtype=np.float32),
            dtype=np.float32
        )


        self._step_count = 0
        self.last_action = 0

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

        # pitch_hist = self._pad_history(self.pitch_action_history)
        # yaw_hist   = self._pad_history(self.yaw_action_history)
        action_hist = self._pad_history(self.action_history)

        norm_beam_E = self.beam_energy_E0 / self.MAX_ENERGY
        norm_coh_E  = self.coherent_edge_Ei / self.MAX_ENERGY
        norm_peak   = peak / self.MAX_ENERGY
        norm_dose   = self.sim.dose.dose / self.MAX_DOSE

        relative_error = abs(peak - self.coherent_edge_Ei) / (self.coherent_edge_Ei + 1e-8)
        return np.array([
            norm_beam_E,
            norm_coh_E,
            norm_peak,
            relative_error,
            # self.sim.goni.return_diamond_pitch(),
            # self.sim.goni.return_diamond_yaw(),
            norm_dose,
            self.orientation_index,
            sign_error,
            *action_hist
            # *pitch_hist,
            # *yaw_hist
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Gym API
    # ----------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        target_peak_random = self.np_random.uniform(low=self.target_edge_low, high=self.target_edge_high)
        self.coherent_edge_Ei = target_peak_random

        sign = self.np_random.choice([-1.0, 1.0])
        magnitude = self.np_random.uniform(low=10.0, high=50.0)
        peak_offset = sign * magnitude
        randomized_peak = self.coherent_edge_Ei + peak_offset

        accumulated_dose = self.sim.dose.dose
        if accumulated_dose > self.MAX_DOSE:
            # reset accumulated dose if it exceeds max (to prevent obs normalization issues)
            accumulated_dose = 0.0

        self.sim = CoherentBremsstrahlungSimulator(
            base_peak_position=randomized_peak,
            dose_slope=self.sim.peak_tracker.dose_slope,
            beam_energy_E0=self.beam_energy_E0,
            coherent_edge_Ei=self.coherent_edge_Ei,
            orientation=self.orientation_label,
            run_period=self.run_period,
            accumulated_dose=accumulated_dose,

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

        self.action_history.clear()
        self.last_action = 0
        self.steps_stable_counter = 0

        return self._get_obs(peak), {}


    def step(self, action):
        self._step_count += 1
        
        action_dir = self._map_action(action)
        if action_dir != 0:
            self.action_history.append(action_dir)


        if self.orientation_index == 0:  # PERP 0/90
            pitch_dir = action_dir
            yaw_dir = 0
        elif self.orientation_index == 1:  # PARA 0/90
            pitch_dir = 0
            yaw_dir = action_dir
        elif self.orientation_index == 2:  # PERP 45/135
            pitch_dir = -action_dir
            yaw_dir = action_dir
        elif self.orientation_index == 3:  # PARA 45/135
            pitch_dir = action_dir
            yaw_dir = action_dir
        else:
            raise ValueError("Invalid orientation index")

        
        dp = pitch_dir * self.pitch_step_deg
        dy = yaw_dir   * self.yaw_step_deg


        # prev_peak = self.sim.peak_tracker.current_peak_position

        delta_c, peak = self.sim.step(
            dpitch_deg=dp,
            dyaw_deg=dy,
            delta_dose=self.dose_per_step
        )

        
        error = peak - self.coherent_edge_Ei
        relative_error = abs(peak - self.coherent_edge_Ei) / (self.coherent_edge_Ei + 1e-8)
        
        #### Reward: minimize distance from coherent edge
        MIN_STEP_MEV = 2.0
        relative_step_size = MIN_STEP_MEV / (self.coherent_edge_Ei + 1e-8)
        # 1. base reward
        reward = -2000.0 * relative_error  # linear penalty based on relative error

        

        sigma = 2.0 * relative_step_size
        gaussian_bonus = 2.0 * np.exp(-(relative_error**2) / (2*(sigma**2)))
        reward += gaussian_bonus


        target_tolerance = 1.5*relative_step_size
        
        if relative_error < target_tolerance:
            self.steps_stable_counter += 1
        #     if action_dir != 0:
        #         reward -= 1.0
        #     else:
        #         reward += 2.0
        
        # else:        
        
        action_penalty_ratio = 0.35
        action_penalty = (pitch_dir**2 + yaw_dir**2)**0.5   # small penalty for taking an action

        reward -= action_penalty_ratio * action_penalty  # always penalize actions to encourage efficiency


        # Termination
        # if relative_error < 0.001:
        #     terminated = True
        #     reward += 200
        # else:
        #     terminated = False


        if self._step_count >= self.max_steps:
            terminated = True
            # reward -= 10
        else:
            terminated = False

        truncated = False

        
        info = {
            "peak": peak,
            "error": error,
            "delta_c": delta_c,
            "dose": self.sim.dose.dose,
            "pitch_deg": self.sim.goni.return_diamond_pitch(),
            "yaw_deg": self.sim.goni.return_diamond_yaw(),
            "orientation": self.orientation_label,
            "episode_stability": 0.0
        }

        if terminated or truncated:
            stability_ratio = self.steps_stable_counter / self._step_count
            info["episode_stability"] = stability_ratio

        return self._get_obs(peak), reward, terminated, truncated, info

    def calculate_reward(self, relative_error):
        a = 1  # Maximum reward
        b = 0.01  # Controls the steepness of the curve
        c = 0.001  # Controls the center point of the curve
        safe_exp = np.clip((relative_error - c) / b, -709, 709)  # Prevent exp overflow
        return a / (1 + np.exp(safe_exp))

    def close(self):
        pass
