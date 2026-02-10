#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import signal
import sys
import time
from typing import Any, Optional, Tuple

# -------------------------------
# EPICS Interface Layer
# -------------------------------
class EpicsClient:
    def __init__(self) -> None:
        try:
            import epics
            self._epics = epics
        except Exception as e:
            print("[ERROR] pyepics not available. Install with: pip install pyepics")
            raise e

    def get(self, pv: str) -> Optional[float]:
        try:
            return self._epics.caget(pv)
        except Exception as e:
            print(f"[ERROR] EPICS get failed for {pv}: {e}")
            return None

    def put(self, pv: str, value: Any) -> bool:
        try:
            return bool(self._epics.caput(pv, value, wait=True))
        except Exception as e:
            print(f"[ERROR] EPICS put failed for {pv}: {e}")
            return False


# -------------------------------
# Configuration & PV Map
# -------------------------------
@dataclasses.dataclass
class PvMap:
    beam_current: str
    beam_lock: str
    gonio_pitch_readback: str
    gonio_pitch_setpoint: str
    gonio_yaw_readback: str
    gonio_yaw_setpoint: str
    cbrem_plane: str   # HD:CBREM:PLANE (PARA=1, PERP=2)
    cbrem_phipol: str  # HD:CBREM:PHIPOL (phi is 0 or 45)
    peak_mev: str     #Coherent peak position in MeV
    dose: str
    
    @staticmethod
    def from_json(path: str) -> "PvMap":
        with open(path, "r") as f:
            obj = json.load(f)
        return PvMap(
            beam_current=obj["beam_current"],
            gonio_pitch_readback=obj["gonio_pitch_readback"],
            gonio_pitch_setpoint=obj["gonio_pitch_setpoint"],
            gonio_yaw_readback=obj["gonio_yaw_readback"],
            gonio_yaw_setpoint=obj["gonio_yaw_setpoint"],
            cbrem_plane=obj["cbrem_plane"],
            cbrem_phipol=obj["cbrem_phipol"],
            peak_mev=obj["peak_mev"],
            dose=obj["dose"],
        )

    @staticmethod
    def example() -> "PvMap":
        return PvMap(
            beam_current="IBCAD00CRCUR6",
            gonio_pitch_readback="HD:GONI:PITCH.RBV",
            gonio_pitch_setpoint="HD:GONI:PITCH",
            gonio_yaw_readback="HD:GONI:YAW.RBV",
            gonio_yaw_setpoint="HD:GONI:YAW",
            cbrem_plane="HD:CBREM:PLANE",
            cbrem_phipol="HD:CBREM:PHIPOL",
            peak_mev="HD:CBREM:EDGE",
            dose="HD:dose",              
        )


# -------------------------------
# Safety & Limits
# -------------------------------
@dataclasses.dataclass
class SafetyLimits:
    min_beam_current: float = 100.0
    pitch_min: float = -10.0
    pitch_max: float = 10.0
    yaw_min: float = -10.0
    yaw_max: float = 10.0
    max_step_per_cycle: float = 0.25


# -------------------------------
# Data Structures
# -------------------------------
@dataclasses.dataclass
class PlantState:
    timestamp: float
    beam_current: Optional[float]
    gonio_pitch: Optional[float]
    gonio_yaw: Optional[float]
    orientation_raw: Any
    orientation_label: Optional[str] 
    peak_mev: Optional[float]
    dose: Optional[float]

@dataclasses.dataclass
class Action:
    target_pitch: Optional[float]
    target_yaw: Optional[float]
    reason: str = ""
    model_used: str = ""
    orientation_label: str = ""

# -------------------------------
# Control Model Interface
# -------------------------------
class ControlModel:
    name = "base"
    def propose_action(self, state: PlantState) -> Action:
        raise NotImplementedError

    def on_action_applied(self, pitch_dir: int, yaw_dir: int) -> None:
        return    

class RLPolicyManager(ControlModel):
    name = "rl_manager"

    ORIENTATIONS = ["PERP 0/90", "PARA 0/90", "PERP 45/135", "PARA 45/135"]

    def __init__(
        self,
        model_paths: Dict[str, str],
        beam_E0_mev: float,
        Ei_mev: float,
        pitch_step_deg: float,
        yaw_step_deg: float,
        max_energy_mev: float = 12000.0,
        max_dose: float = 100.0,
        history_len: int = 5,
        deterministic: bool = True,
    ):
        # store physics params
        self.beam_E0 = float(beam_E0_mev)
        self.Ei = float(Ei_mev)
        self.pitch_step = float(pitch_step_deg)
        self.yaw_step = float(yaw_step_deg)
        self.MAX_ENERGY = float(max_energy_mev)
        self.MAX_DOSE = float(max_dose)
        self.deterministic = bool(deterministic)

        # history as in training obs
        self.pitch_hist = deque([0.0] * history_len, maxlen=history_len)
        self.yaw_hist = deque([0.0] * history_len, maxlen=history_len)

        # load models
        self.models: Dict[str, Any] = {}
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            raise RuntimeError("stable_baselines3 is required to load PPO .zip models") from e

        for ori, path in model_paths.items():
            if path and os.path.exists(path):
                try:
                    self.models[ori] = PPO.load(path, device="cpu")
                    print(f"[INFO] Loaded model for {ori} from {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load model for {ori} at {path}: {e}")
                    self.models[ori] = None
            else:
                print(f"[WARN] Missing model for {ori}: {path}")
                self.models[ori] = None

    @staticmethod
    def _map_action(a: int) -> int:
        return int(a) - 1  # {0,1,2} -> {-1,0,1}

    def _sign_error(self, peak: float) -> float:
        # +1 if peak > Ei (above nominal edge), -1 otherwise
        return 1.0 if peak > self.Ei else -1.0

    def _orientation_index(self, label: str) -> int:
        return self.ORIENTATIONS.index(label)

    def _build_obs(self, s: PlantState, ori_label: str) -> List[float]:
        # Normalize beam & Ei to MAX_ENERGY, dose to MAX_DOSE
        norm_beam = self.beam_E0 / self.MAX_ENERGY
        norm_coh = self.Ei / self.MAX_ENERGY
        norm_dose = float(s.dose) / self.MAX_DOSE if s.dose is not None else 0.0
        ori_idx = float(self._orientation_index(ori_label))

        sign_err = self._sign_error(float(s.peak_mev)) if s.peak_mev is not None else 0.0

        obs = [
            float(norm_beam),
            float(norm_coh),
            float(norm_dose),
            float(ori_idx),
            float(sign_err),
            *list(self.pitch_hist),
            *list(self.yaw_hist),
        ]
        return obs

    def propose_action(self, s: PlantState) -> Action:
        if s.orientation_label is None:
            return Action(None, None, reason="rl_missing_orientation")

        ori = s.orientation_label
        if ori not in self.models:
            return Action(None, None, reason="rl_bad_orientation", model_used="none", orientation_label=ori)
        model = self.models[ori]
        if model is None:
            return Action(None, None, reason="rl_no_model", model_used="none", orientation_label=ori)

        # Require the PVs that agent expects
        if s.peak_mev is None or s.dose is None or s.gonio_pitch is None or s.gonio_yaw is None:
            return Action(None, None, reason="rl_missing_inputs", model_used=self.safe_label(ori), orientation_label=ori)

        try:
            import numpy as np

            obs = np.array(self._build_obs(s, ori), dtype=np.float32)
            action_raw, _ = model.predict(obs, deterministic=self.deterministic)

            # Expect MultiDiscrete([3,3])
            if hasattr(action_raw, "__len__"):
                a_p = int(action_raw[0])
                a_y = int(action_raw[1]) if len(action_raw) > 1 else 1
            else:
                a_p, a_y = 1, 1

            p_dir = self._map_action(a_p)
            y_dir = self._map_action(a_y)

            target_pitch = float(s.gonio_pitch) + p_dir * self.pitch_step
            target_yaw = float(s.gonio_yaw) + y_dir * self.yaw_step

            return Action(target_pitch, target_yaw, reason="rl_policy", model_used=self.safe_label(ori), orientation_label=ori)
        except Exception as e:
            print(f"[ERROR] RL inference failed for {ori}: {e}")
            return Action(None, None, reason="rl_inference_error", model_used=self.safe_label(ori), orientation_label=ori)

    def on_action_applied(self, pitch_dir: int, yaw_dir: int) -> None:
        # mirror training env: append ±1 or 0
        self.pitch_hist.append(float(pitch_dir))
        self.yaw_hist.append(float(yaw_dir))

    @staticmethod
    def safe_label(label: str) -> str:
        return label.replace(" ", "_").replace("/", "-")


# -------------------------------
# Logger
# -------------------------------
class DecisionLogger:
    def __init__(self, logdir: str, no_act: bool) -> None:
        os.makedirs(logdir, exist_ok=True)
        self.no_act = no_act
        self.path = os.path.join(logdir, f"decisions_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_file()
    def _init_file(self) -> None:
        header = ["iso_time","beam_current","beam_lock","gonio_pitch","gonio_yaw","proposed_pitch","proposed_yaw","applied_pitch","applied_yaw","reason","status","no_act_mode"]
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    def log(self, state: PlantState, proposed_pitch: Optional[float], proposed_yaw: Optional[float], applied_pitch: Optional[float], applied_yaw: Optional[float], reason: str, status: str) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                dt.datetime.fromtimestamp(state.timestamp).isoformat(),
                _fmt(state.beam_current),
                _fmt(state.beam_lock),
                _fmt(state.gonio_pitch),
                _fmt(state.gonio_yaw),
                _fmt(proposed_pitch),
                _fmt(proposed_yaw),
                _fmt(applied_pitch),
                _fmt(applied_yaw),
                reason,
                status,
                str(self.no_act),
            ])

def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.6f}"


# -------------------------------
# Control Loop
# -------------------------------
class ControlLoop:
    ORI_MAP = {
        (1, 0): "PARA 0/90",
        (1, 45): "PARA 45/135",
        (2, 0): "PERP 0/90",
        (2, 45): "PERP 45/135",
    }

    def __init__(self, client: EpicsClient, pv: PvMap, limits: SafetyLimits, model: ControlModel,
                 interval_s: float, logger: DecisionLogger, no_act: bool = False):
        self.client = client
        self.pv = pv
        self.limits = limits
        self.model = model
        self.dt = max(1.0, float(interval_s))
        self.logger = logger
        self.no_act = no_act
        self._stop = False

        self._last_applied_pitch: Optional[float] = None
        self._last_applied_yaw: Optional[float] = None

    def run(self) -> None:
        print(f"[INFO] Starting control loop; interval={self.dt}s; model={self.model.name}; no_act={self.no_act}")
        self._install_signal_handlers()

        while not self._stop:
            t0 = time.time()
            state = self._read_state()

            status = "ok"
            proposed_pitch = proposed_yaw = None
            applied_pitch = applied_yaw = None
            action = Action(None, None)

            try:
                if not self._safe_to_act(state):
                    status = "blocked_by_safety_or_missing_signals"
                    action = Action(None, None, reason="safety_or_missing_pvs")
                else:
                    action = self.model.propose_action(state)
                    proposed_pitch, proposed_yaw = action.target_pitch, action.target_yaw

                    if proposed_pitch is None or proposed_yaw is None:
                        status = "no_action"
                    else:
                        # clamp to hard limits + per-cycle max step
                        tp = self._apply_limits(state.gonio_pitch, proposed_pitch, self.limits.pitch_min, self.limits.pitch_max)
                        ty = self._apply_limits(state.gonio_yaw, proposed_yaw, self.limits.yaw_min, self.limits.yaw_max)

                        # extra absolute delta guard
                        if abs(tp - state.gonio_pitch) > self.limits.max_abs_target_delta or abs(ty - state.gonio_yaw) > self.limits.max_abs_target_delta:
                            status = "blocked_large_jump"
                        else:
                            if not self.no_act:
                                ok1 = self.client.put(self.pv.gonio_pitch_setpoint, tp)
                                ok2 = self.client.put(self.pv.gonio_yaw_setpoint, ty)
                                applied_pitch = tp if ok1 else None
                                applied_yaw = ty if ok2 else None
                                status = "applied" if (ok1 and ok2) else "caput_failed"
                            else:
                                status = "no_act"

                            # record the discrete directions actually executed for model history
                            pitch_dir = self._dir_from_delta(state.gonio_pitch, tp)
                            yaw_dir = self._dir_from_delta(state.gonio_yaw, ty)
                            self.model.on_action_applied(pitch_dir, yaw_dir)

                            self._last_applied_pitch = tp
                            self._last_applied_yaw = ty

            except Exception as e:
                print(f"[ERROR] Control step failed: {e}")
                status = "error"

            # log the decision (including Ei in the logger)
            self.logger.log(state, proposed_pitch, proposed_yaw, applied_pitch, applied_yaw, action, status)

            elapsed = time.time() - t0
            remaining = self.dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print("[INFO] Control loop stopped.")

    def _read_state(self) -> PlantState:
        ts = time.time()

        beam_current = self._to_float(self.client.get(self.pv.beam_current))
        pitch = self._to_float(self.client.get(self.pv.gonio_pitch_readback))
        yaw = self._to_float(self.client.get(self.pv.gonio_yaw_readback))

        plane_raw = self.client.get(self.pv.cbrem_plane)
        phipol_raw = self.client.get(self.pv.cbrem_phipol)
        orientation_label = self._derive_orientation_from_plane_phi(plane_raw, phipol_raw)

        peak_mev = self._to_float(self.client.get(self.pv.peak_mev))
        dose_val = self._to_float(self.client.get(self.pv.dose))

        return PlantState(
            timestamp=ts,
            beam_current=beam_current,
            gonio_pitch=pitch,
            gonio_yaw=yaw,
            cbrem_plane_raw=plane_raw,
            cbrem_phipol_raw=phipol_raw,
            orientation_label=orientation_label,
            peak_mev=peak_mev,
            dose=dose_val,
        )

    def _derive_orientation_from_plane_phi(self, plane_raw: Any, phi_raw: Any) -> Optional[str]:
        # Plane: PARA=1, PERP=2
        try:
            plane = int(float(plane_raw))
        except Exception:
            return None
        try:
            phi = int(float(phi_raw))
        except Exception:
            return None

        return self.ORI_MAP.get((plane, phi))

    def _safe_to_act(self, state: PlantState) -> bool:
        # Safety checks without beam_lock
        if state.beam_current is None or state.beam_current < self.limits.min_beam_current:
            return False
        if state.orientation_label is None:
            return False
        if state.peak_mev is None or state.dose is None:
            return False
        if state.gonio_pitch is None or state.gonio_yaw is None:
            return False
        return True



# -------------------------------
# CLI & Main
# -------------------------------
def load_pvmap(path: Optional[str]) -> PvMap:
    if path and os.path.exists(path):
        print(f"[INFO] Loading PV map from {path}"); return PvMap.from_json(path)
    print("[WARN] Using example PV map – replace with Hall D PVs."); return PvMap.example()

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GlueX goniometer control (RL pitch+yaw, pyepics)")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--pvmap", type=str, default=None)
    p.add_argument("--logdir", type=str, default="./logs")
    p.add_argument("--rl-checkpoint", type=str, default=None)
    p.add_argument("--rl-scale", type=float, default=0.1)
    p.add_argument("--no-act", action="store_true", help="Do not send actions to EPICS (log only)")
    return p.parse_args(argv)

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    pvmap = load_pvmap(args.pvmap); limits = SafetyLimits(); logger = DecisionLogger(args.logdir, args.no_act)
    client = EpicsClient()
    model = RLPolicyModel(checkpoint_path=args.rl_checkpoint, scale=args.rl_scale)
    loop = ControlLoop(client, pvmap, limits, model, args.interval, logger, no_act=args.no_act)
    try:
        loop.run(); return 0
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user."); return 130

if __name__ == "__main__":
    sys.exit(main())
