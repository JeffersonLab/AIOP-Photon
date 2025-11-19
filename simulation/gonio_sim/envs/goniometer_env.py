from dataclasses import dataclass
import numpy as np
import yaml

# Import physics primitives from the gonio_sim package
from gonio_sim.physics import (
    AxisBacklashState,
    step_with_backlash,
    apply_wobble,
    compute_peak_energy,
)
from gonio_sim.utils.offsets import sample_offsets
from damage_model import DiamondDamageModel
from spotfinder import make_beamtilt, snap_crystal_orientation


@dataclass
class EnvConfig:
    edge: float = 8.6
    config: str = "PARA" # PARA or PERP
    phi: str = "0/90" # 0/90 or 45/135
    backlash_n: float = 2.0
    step_mrad: float = 0.01745


class GoniometerEnv:
    """Stateful one-step API for controllers and batch drivers."""
    def __init__(self, cfg: EnvConfig | None = None, cfg_file: str | None = None):
        if cfg is None:
            cfg = EnvConfig()
        self.cfg = cfg


        # Load defaults from YAML
        with open(__file__.replace("goniometer_env.py", "../configs/default.yaml"), "r") as f:
            base_args = yaml.safe_load(f)

        self.base_args = {
            **base_args,
            "snapedge": cfg.edge,
        }

        phi_map = {("PARA", "0/90"): 0, ("PERP", "0/90"): 90,
                   ("PARA", "45/135"): 135, ("PERP", "45/135"): 45}
        self.base_args["phideg"] = phi_map[(cfg.config, cfg.phi)]
        self.base_args["snapact"] = "snap_para" if cfg.config == "PARA" else "snap_perp"

        self.damage = DiamondDamageModel()
        self.offsets = sample_offsets()
        self.dose = 0.0

        make_beamtilt(self.base_args)
        status, output = snap_crystal_orientation(self.base_args)
        thetah_0, thetav_0 = map(lambda s: float(s.strip()), output[0].decode().split(","))

        self.yaw_state = AxisBacklashState(pos_true=thetah_0, pos_readback=thetah_0)
        self.pitch_state = AxisBacklashState(pos_true=thetav_0, pos_readback=thetav_0)

        w = base_args.get("wobble", {})
        self.yaw_wobble_amp = w.get("yaw_amp", 0.0004)
        self.yaw_wobble_period = w.get("yaw_period", 0.05)
        self.pitch_wobble_amp = w.get("pitch_amp", 0.0006)
        self.pitch_wobble_period = w.get("pitch_period", 0.01)
        self.yaw_phase = np.random.uniform(0, 2*np.pi)
        self.pitch_phase = np.random.uniform(0, 2*np.pi)


    def step(self, pitch_delta_deg: float, yaw_delta_deg: float) -> float:
        # Convert degrees to mrad
        dp = pitch_delta_deg * 1e3
        dy = yaw_delta_deg * 1e3

        yaw_target = self.yaw_state.pos_true + dy
        pitch_target = self.pitch_state.pos_true + dp

        yaw_true_nowobble, yaw_readback = step_with_backlash(
            yaw_target, self.cfg.step_mrad, self.cfg.backlash_n, self.yaw_state)
        pitch_true_nowobble, pitch_readback = step_with_backlash(
            pitch_target, self.cfg.step_mrad, self.cfg.backlash_n, self.pitch_state)

        yaw_true = apply_wobble(yaw_true_nowobble, self.yaw_wobble_amp, self.yaw_wobble_period, self.yaw_phase)
        pitch_true = apply_wobble(pitch_true_nowobble, self.pitch_wobble_amp, self.pitch_wobble_period, self.pitch_phase)


        beam_delh = np.random.normal(0.0, 0.002)
        beam_delv = np.random.normal(0.0, 0.001)


        params = (0, 0,
                  yaw_true_nowobble, pitch_true_nowobble,
                  yaw_true, pitch_true,
                  yaw_readback, pitch_readback,
                  beam_delh, beam_delv,
                  self.base_args, self.dose, self.damage, self.offsets)

        # Index 12 is coherent peak energy (GeV)
        return compute_peak_energy(params)[12]


    def reset(self):
        """Optional: re-sample offsets & phases (useful for RL)."""
        self.offsets = sample_offsets()
        self.yaw_phase = np.random.uniform(0, 2*np.pi)
        self.pitch_phase = np.random.uniform(0, 2*np.pi)
