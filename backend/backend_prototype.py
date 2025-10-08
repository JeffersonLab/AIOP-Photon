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

    @staticmethod
    def from_json(path: str) -> "PvMap":
        with open(path, "r") as f:
            obj = json.load(f)
        return PvMap(
            beam_current=obj["beam_current"],
            beam_lock=obj["beam_lock"],
            gonio_pitch_readback=obj["gonio_pitch_readback"],
            gonio_pitch_setpoint=obj["gonio_pitch_setpoint"],
            gonio_yaw_readback=obj["gonio_yaw_readback"],
            gonio_yaw_setpoint=obj["gonio_yaw_setpoint"],
        )

    @staticmethod
    def example() -> "PvMap":
        return PvMap(
            beam_current="beam:current",
            beam_lock="beam:lock",
            gonio_pitch_readback="gonio:pitch",
            gonio_pitch_setpoint="gonio:pitch_set",
            gonio_yaw_readback="gonio:yaw",
            gonio_yaw_setpoint="gonio:yaw_set",
        )


# -------------------------------
# Safety & Limits
# -------------------------------
@dataclasses.dataclass
class SafetyLimits:
    min_beam_current: float = 10.0
    require_beam_lock: bool = True
    pitch_min: float = -5.0
    pitch_max: float = 5.0
    yaw_min: float = -5.0
    yaw_max: float = 5.0
    max_step_per_cycle: float = 0.25


# -------------------------------
# Data Structures
# -------------------------------
@dataclasses.dataclass
class PlantState:
    timestamp: float
    beam_current: Optional[float]
    beam_lock: Optional[float]
    gonio_pitch: Optional[float]
    gonio_yaw: Optional[float]


@dataclasses.dataclass
class Action:
    target_pitch: Optional[float]
    target_yaw: Optional[float]
    reason: str = ""


# -------------------------------
# RL Policy Model
# -------------------------------
class ControlModel:
    name = "base"
    def propose_action(self, state: PlantState) -> Action:
        raise NotImplementedError


class RLPolicyModel(ControlModel):
    name = "rl"

    def __init__(self, checkpoint_path: Optional[str] = None, scale: float = 0.1) -> None:
        self.checkpoint_path = checkpoint_path
        self.scale = float(scale)
        self._torch = None
        self._policy = None
        self._load_policy()

    def _load_policy(self) -> None:
        if self.checkpoint_path:
            try:
                import torch  # type: ignore
                self._torch = torch
                self._policy = torch.jit.load(self.checkpoint_path) if self.checkpoint_path.endswith(".pt") else torch.load(self.checkpoint_path)
                self._policy.eval()
                return
            except Exception as e:
                print(f"[WARN] Failed to load PyTorch checkpoint '{self.checkpoint_path}': {e}\nUsing built-in linear policy instead.")
        self._policy = None

    def _forward(self, bc: float, lock: float, pitch: float, yaw: float) -> Tuple[float, float]:
        bc_norm = bc / 100.0
        lk = 1.0 if lock >= 0.5 else 0.0
        pitch_norm = pitch / 5.0
        yaw_norm = yaw / 5.0
        if self._policy is not None and self._torch is not None:
            with self._torch.no_grad():
                inp = self._torch.tensor([[bc_norm, lk, pitch_norm, yaw_norm]], dtype=self._torch.float32)
                out = self._policy(inp).squeeze().tolist()
                if isinstance(out, float):
                    out = [out, 0.0]
        else:
            out_pitch = 0.3 * bc_norm + 0.2 * lk - 0.3 * pitch_norm
            out_yaw = -0.2 * bc_norm + 0.1 * lk - 0.3 * yaw_norm
            out = [out_pitch, out_yaw]
        delta_pitch = math.tanh(out[0]) * self.scale
        delta_yaw = math.tanh(out[1]) * self.scale
        return delta_pitch, delta_yaw

    def propose_action(self, state: PlantState) -> Action:
        if state.gonio_pitch is None or state.gonio_yaw is None:
            return Action(None, None, reason="rl_missing_angles")
        if state.beam_current is None or state.beam_lock is None:
            return Action(None, None, reason="rl_missing_inputs")
        d_pitch, d_yaw = self._forward(state.beam_current, state.beam_lock, state.gonio_pitch, state.gonio_yaw)
        target_pitch = state.gonio_pitch + d_pitch
        target_yaw = state.gonio_yaw + d_yaw
        return Action(target_pitch=target_pitch, target_yaw=target_yaw, reason="rl_policy")


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
    def __init__(self, client: EpicsClient, pvmap: PvMap, limits: SafetyLimits, model: ControlModel, interval_s: float, logger: DecisionLogger, no_act: bool = False) -> None:
        self.client = client; self.pv = pvmap; self.limits = limits
        self.model = model; self.dt = max(1.0, float(interval_s))
        self.log = logger; self._stop = False
        self.no_act = no_act

    def run(self) -> None:
        print(f"[INFO] Starting control loop; interval={self.dt}s; model={self.model.name}; no_act={self.no_act}")
        self._install_signal_handlers()
        while not self._stop:
            start = time.time(); state = self._read_state()
            status, applied_pitch, applied_yaw, proposed_pitch, proposed_yaw, reason = "ok", None, None, None, None, ""
            try:
                if not self._safe_to_act(state):
                    status, reason = "blocked_by_safety", "safety_check_failed"
                else:
                    action = self.model.propose_action(state)
                    proposed_pitch, proposed_yaw, reason = action.target_pitch, action.target_yaw, action.reason
                    if proposed_pitch is not None and proposed_yaw is not None and state.gonio_pitch is not None and state.gonio_yaw is not None:
                        target_pitch = self._apply_safety(state.gonio_pitch, proposed_pitch, self.limits.pitch_min, self.limits.pitch_max)
                        target_yaw = self._apply_safety(state.gonio_yaw, proposed_yaw, self.limits.yaw_min, self.limits.yaw_max)
                        if not self.no_act:
                            ok1 = self.client.put(self.pv.gonio_pitch_setpoint, target_pitch)
                            ok2 = self.client.put(self.pv.gonio_yaw_setpoint, target_yaw)
                            applied_pitch = target_pitch if ok1 else None
                            applied_yaw = target_yaw if ok2 else None
                            status = "applied" if (ok1 and ok2) else "caput_failed"
                        else:
                            applied_pitch, applied_yaw = None, None
                            status = "no_act"
                    else:
                        status = "no_action"
            except Exception as e:
                print(f"[ERROR] Control step failed: {e}"); status = "error"
            self.log.log(state, proposed_pitch, proposed_yaw, applied_pitch, applied_yaw, reason, status)
            elapsed = time.time() - start; remaining = self.dt - elapsed
            if remaining > 0: time.sleep(remaining)
        print("[INFO] Control loop stopped.")

    def _read_state(self) -> PlantState:
        ts = time.time()
        return PlantState(
            ts,
            self.client.get(self.pv.beam_current),
            self.client.get(self.pv.beam_lock),
            self.client.get(self.pv.gonio_pitch_readback),
            self.client.get(self.pv.gonio_yaw_readback),
        )

    def _safe_to_act(self, state: PlantState) -> bool:
        if self.limits.require_beam_lock and (state.beam_lock is None or state.beam_lock < 0.5): return False
        if state.beam_current is None or state.beam_current < self.limits.min_beam_current: return False
        return state.gonio_pitch is not None and state.gonio_yaw is not None

    def _apply_safety(self, current: float, proposed: float, amin: float, amax: float) -> float:
        bounded = max(amin, min(amax, proposed))
        dtheta = max(-self.limits.max_step_per_cycle, min(self.limits.max_step_per_cycle, bounded - current))
        return current + dtheta

    def _install_signal_handlers(self) -> None:
        def _stop_handler(signum, frame):
            print(f"[INFO] Received signal {signum}; stopping at next cycle..."); self._stop = True
        try:
            signal.signal(signal.SIGINT, _stop_handler); signal.signal(signal.SIGTERM, _stop_handler)
        except Exception: pass


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
