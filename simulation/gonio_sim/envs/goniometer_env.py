from dataclasses import dataclass
import numpy as np
import yaml

# Physics primitives
from gonio_sim.physics import (
    AxisBacklashState,
    step_with_backlash,
    apply_wobble,
    compute_peak_energy,
)
from gonio_sim.utils.offsets import sample_offsets
from gonio_sim.physics.damage_model import DiamondDamageModel
from spotfinder import make_beamtilt, snap_crystal_orientation


@dataclass
class EnvConfig:
    edge: float = 8.6
    config: str = "PARA"   # PARA or PERP
    phi: str = "0/90"       # 0/90 or 45/135
    backlash_n: float = 2.0
    step_mrad: float = 0.01745   # movement per motor step


class GoniometerEnv:
    """
    High-level simulation environment for beam–diamond–goniometer behavior.
    Contains:
      - backlash + wobble models
      - correct Cobrems geometry
      - spatial dose map (FULLY FUNCTIONAL)
      - radiation-damage model

    Works with RL, scanning scripts, and plotting utilities.
    """

    def __init__(self, cfg: EnvConfig | None = None, cfg_file: str | None = None):
        if cfg is None:
            cfg = EnvConfig()
        self.cfg = cfg

        # Load defaults from YAML
        with open(__file__.replace("goniometer_env.py", "../configs/default.yaml"), "r") as f:
            base_args = yaml.safe_load(f)

        # Merge with config
        self.base_args = {**base_args, "snapedge": cfg.edge}

        # Mapping: (config, phi_set) → phideg
        phi_map = {
            ("PARA", "0/90"):   0,
            ("PERP", "0/90"):  90,
            ("PARA", "45/135"): 135,
            ("PERP", "45/135"): 45,
        }

        self.base_args["phideg"] = phi_map[(cfg.config, cfg.phi)]
        self.base_args["snapact"] = "snap_para" if cfg.config == "PARA" else "snap_perp"

        # Damage model + offsets
        self.damage = DiamondDamageModel()
        self.offsets = sample_offsets()

        # GLOBAL scalar dose (for reporting only)
        self.dose = 0.0

        # ------------------------------------------------------------
        #  SPATIAL DOSE MAP (FULL 70×70 mm RADIATOR)
        # ------------------------------------------------------------
        Nx = Ny = 141          # 0.5 mm resolution
        half_size = 35.0       # ±35 mm = 70 mm total

        self.grid_x = np.linspace(-half_size, half_size, Nx)
        self.grid_y = np.linspace(-half_size, half_size, Ny)

        self.dose_map = np.zeros((Nx, Ny), dtype=float)

        # How much dose is deposited per irradiation step
        self.dose_per_step = 1.0

        # ------------------------------------------------------------
        # Compute beam-topography alignment
        # ------------------------------------------------------------
        make_beamtilt(self.base_args)
        status, output = snap_crystal_orientation(self.base_args)

        thetah_0, thetav_0 = map(lambda s: float(s.strip()),
                                 output[0].decode().split(","))

        # Backlash states
        self.yaw_state = AxisBacklashState(
            pos_true=thetah_0, pos_readback=thetah_0)
        self.pitch_state = AxisBacklashState(
            pos_true=thetav_0, pos_readback=thetav_0)

        # Wobble parameters
        w = base_args.get("wobble", {})
        self.yaw_wobble_amp = w.get("yaw_amp", 0.0004)
        self.yaw_wobble_period = w.get("yaw_period", 0.05)
        self.pitch_wobble_amp = w.get("pitch_amp", 0.0006)
        self.pitch_wobble_period = w.get("pitch_period", 0.01)

        self.yaw_phase = np.random.uniform(0, 2*np.pi)
        self.pitch_phase = np.random.uniform(0, 2*np.pi)

    # ----------------------------------------------------------------------
    # FAST DOSE DEPOSITION HELPERS
    # ----------------------------------------------------------------------

    def deposit_dose(self, x_mm, y_mm):
        """
        Deposit dose at EXACT location (nearest neighbor).
        Used for ultra-fast irradiation without running any physics.
        """
        ix = int(np.argmin(np.abs(self.grid_x - x_mm)))
        iy = int(np.argmin(np.abs(self.grid_y - y_mm)))
        self.dose_map[ix, iy] += self.dose_per_step

    def deposit_dose_gaussian(self, x_mm, y_mm, sigma=0.3):
        """
        Deposit dose with a small Gaussian beam PSF.
        sigma in mm (≈ RMS of beam footprint).
        """
        dx = self.grid_x[:, None] - x_mm
        dy = self.grid_y[None, :] - y_mm
        r2 = dx*dx + dy*dy

        kernel = np.exp(-0.5 * r2 / (sigma*sigma))
        self.dose_map += kernel * self.dose_per_step

    # ----------------------------------------------------------------------
    # Local dose lookup for damage model
    # ----------------------------------------------------------------------

    def get_local_dose(self, x_mm, y_mm):
        """
        Returns ONLY the scalar dose at (x_mm, y_mm).
        This must be used inside step() so damage.apply()
        receives different doses for different beam positions.
        """
        ix = int(np.argmin(np.abs(self.grid_x - x_mm)))
        iy = int(np.argmin(np.abs(self.grid_y - y_mm)))
        return self.dose_map[ix, iy]

    # ----------------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------------

    def reset(self):
        self.offsets = sample_offsets()
        self.yaw_phase = np.random.uniform(0, 2*np.pi)
        self.pitch_phase = np.random.uniform(0, 2*np.pi)
        self.dose_map[:] = 0.0
        self.dose = 0.0

    # ----------------------------------------------------------------------
    # MAIN STEP FUNCTION (ONE MOVE + ONE PHYSICS EVAL)
    # ----------------------------------------------------------------------

    def step(self, pitch_delta_deg: float, yaw_delta_deg: float) -> float:
        """
        Perform one 'move' of the goniometer and evaluate the physics.

        NOTE:
            * BEFORE calling compute_peak_energy() we lookup local dose.
            * This ensures spectra differ by position due to damage.
        """

        # Convert degrees → mrad (your simulation units)
        dp = pitch_delta_deg * 1e3
        dy = yaw_delta_deg * 1e3

        # Apply backlash model
        yaw_target = self.yaw_state.pos_true + dy
        pitch_target = self.pitch_state.pos_true + dp

        yaw_true_nowobble, yaw_readback = step_with_backlash(
            yaw_target, self.cfg.step_mrad, self.cfg.backlash_n, self.yaw_state
        )
        pitch_true_nowobble, pitch_readback = step_with_backlash(
            pitch_target, self.cfg.step_mrad, self.cfg.backlash_n, self.pitch_state
        )

        # Apply mechanical wobble
        yaw_true = apply_wobble(
            yaw_true_nowobble, self.yaw_wobble_amp, self.yaw_wobble_period, self.yaw_phase
        )
        pitch_true = apply_wobble(
            pitch_true_nowobble, self.pitch_wobble_amp, self.pitch_wobble_period, self.pitch_phase
        )

        # Instrumental beam jitter
        beam_delh = np.random.normal(0.0, 0.002)
        beam_delv = np.random.normal(0.0, 0.001)

        # ------------------------------------------------------------
        # CRITICAL: Update dose from dose_map
        # ------------------------------------------------------------
        x0 = self.base_args["xoffset"]
        y0 = self.base_args["yoffset"]
        self.dose = self.get_local_dose(x0, y0)

        # Build parameter tuple for physics engine
        params = (
            0, 0,
            yaw_true_nowobble, pitch_true_nowobble,
            yaw_true, pitch_true,
            yaw_readback, pitch_readback,
            beam_delh, beam_delv,
            self.base_args, self.dose, self.damage, self.offsets,
        )

        # Evaluate physics → peak energy
        result = compute_peak_energy(params)

        # Deposit new dose only during irradiation phase
        if self.dose_per_step != 0.0:
            self.deposit_dose_gaussian(x0, y0, sigma=0.3)

        return result
