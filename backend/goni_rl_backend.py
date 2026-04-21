#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import epics
from jlab_archiver_client import Point, PointQuery
from stable_baselines3 import PPO


# ----------------------------------------------------------------------
# Constants matching the uploaded environment/simulator
# ----------------------------------------------------------------------

ORIENTATIONS = [
    "PERP 0/90",
    "PARA 0/90",
    "PERP 45/135",
    "PARA 45/135",
]

ORIENTATION_TO_PHI = {
    "PARA 0/90": 0.0,
    "PERP 0/90": 90.0,
    "PARA 45/135": 45.0,
    "PERP 45/135": 135.0,
}

MAX_ENERGY = 12000.0
MAX_DOSE = 500.0

STATUS_INIT = 0
STATUS_RUNNING = 1
STATUS_ERROR = 2
STATUS_STOPPED = 3
STATUS_INHIBITED = 4


# ----------------------------------------------------------------------
# PV configuration
# ----------------------------------------------------------------------

MYA_PVS = {
    "beam_energy_E0": "HALLD:p",
    "nominal_edge": "HD:CBREM:REQ_EDGE",
    "measured_edge": "HD:CBREM:EDGE",
    "beam_current": "IBCAD00CRCUR6",
    "plane": "HD:CBREM:PLANE",           # PARA = 1, PERP = 2
    "phipol": "HD:CBREM:PHIPOL",         # 0 or 45
    "radiator_name": "HD:GONI:RADIATOR_NAME",
}

WRITE_PVS = {
    "delta_pitch_req": "HD:CBREM:DELTA_PITCH_REQ_AI",
    "delta_yaw_req": "HD:CBREM:DELTA_YAW_REQ_AI",
    "delta_c_req": "HD:CBREM:DELTA_C-ANGLE_REQ_AI",
    "heartbeat": "HD:CBREM:AI_HEARTBEAT",
    "status": "HD:CBREM:AI_STATUS",
}

DIAMOND_NAME_RE = re.compile(r"JD\d{2}-\d{3}")


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------

@dataclass
class LiveState:
    beam_energy_E0: float
    coherent_edge_Ei: float
    peak_energy: float
    dose: float
    beam_current: float
    orientation_index: int
    radiator_name: str
    beam_tilt_pitch_deg: float = 0.0
    beam_tilt_yaw_deg: float = 0.0


class ActionHistory:
    """
    Matches rl_env.py behavior:
    - only nonzero pitch/yaw actions are stored
    - only the last 5 are kept
    """

    def __init__(self) -> None:
        self.pitch = deque(maxlen=5)
        self.yaw = deque(maxlen=5)

    @staticmethod
    def _pad(history: deque, length: int = 5) -> list:
        return [0.0] * (length - len(history)) + list(history)

    def append(self, pitch_dir: int, yaw_dir: int) -> None:
        if pitch_dir != 0:
            self.pitch.append(pitch_dir)
        if yaw_dir != 0:
            self.yaw.append(yaw_dir)

    def clear(self) -> None:
        self.pitch.clear()
        self.yaw.clear()

    def pitch_padded(self) -> list:
        return self._pad(self.pitch)

    def yaw_padded(self) -> list:
        return self._pad(self.yaw)


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------

def parse_timestamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def format_age_hours(reference_time: datetime, value_time: datetime) -> float:
    return (reference_time - value_time).total_seconds() / 3600.0


def is_diamond_radiator_name(name: str) -> bool:
    if not name:
        return False

    if isinstance(name, (list, tuple)):
        text = " ".join(str(x) for x in name)
    else:
        text = str(name)

    return DIAMOND_NAME_RE.search(text) is not None

def beam_current_ok(beam_current: float, min_beam_current: float) -> bool:
    return beam_current >= min_beam_current

# ----------------------------------------------------------------------
# MYA helpers
# ----------------------------------------------------------------------

def read_mya_point_exact(
    channel: str,
    when: Optional[datetime] = None,
) -> Tuple[Optional[float], Optional[str], dict]:
    """
    Query MYA at a specific time and return:
        (value_or_none, timestamp_string_or_none, raw_event)
    """
    query = PointQuery(
        channel=channel,
        time=when or datetime.now(),
    )
    point = Point(query)
    point.run()

    event = point.event or {}
    data = event.get("data", {})

    if "v" in data:
        return float(data["v"]), data.get("d"), event

    return None, data.get("d"), event


def read_mya_string_point_exact(
    channel: str,
    when: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str], dict]:
    """
    Query MYA at a specific time and return:
        (string_value_or_none, timestamp_string_or_none, raw_event)
    """
    query = PointQuery(
        channel=channel,
        time=when or datetime.now(),
    )
    point = Point(query)
    point.run()

    event = point.event or {}
    data = event.get("data", {})

    if "v" in data:
        return str(data["v"]).strip(), data.get("d"), event

    return None, data.get("d"), event


def find_last_valid_mya_value(
    channel: str,
    start_time: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[float, str, dict]:
    """
    Search backward for the most recent archived numeric value with a usable 'v' field.
    Returns:
        (value, timestamp_string, raw_event)
    """
    base_time = start_time or datetime.now()
    candidate_times = [
        base_time,
        base_time - timedelta(minutes=1),
        base_time - timedelta(minutes=5),
        base_time - timedelta(minutes=15),
        base_time - timedelta(hours=1),
        base_time - timedelta(hours=6),
        base_time - timedelta(days=1),
        base_time - timedelta(days=3),
        base_time - timedelta(days=7),
        base_time - timedelta(days=14),
        base_time - timedelta(days=max_lookback_days),
    ]

    last_event = None

    for t in candidate_times:
        value, ts_str, event = read_mya_point_exact(channel, when=t)
        last_event = event
        if value is not None and ts_str is not None:
            return value, ts_str, event

    raise RuntimeError(
        "No usable MYA point returned for {0}. Last event: {1!r}".format(channel, last_event)
    )


def find_last_valid_mya_string_value(
    channel: str,
    start_time: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[str, str, dict]:
    """
    Search backward for the most recent archived string value with a usable 'v' field.
    Returns:
        (value, timestamp_string, raw_event)
    """
    base_time = start_time or datetime.now()
    candidate_times = [
        base_time,
        base_time - timedelta(minutes=1),
        base_time - timedelta(minutes=5),
        base_time - timedelta(minutes=15),
        base_time - timedelta(hours=1),
        base_time - timedelta(hours=6),
        base_time - timedelta(days=1),
        base_time - timedelta(days=3),
        base_time - timedelta(days=7),
        base_time - timedelta(days=14),
        base_time - timedelta(days=max_lookback_days),
    ]

    last_event = None

    for t in candidate_times:
        value, ts_str, event = read_mya_string_point_exact(channel, when=t)
        last_event = event
        if value is not None and ts_str is not None:
            return value, ts_str, event

    raise RuntimeError(
        "No usable MYA string point returned for {0}. Last event: {1!r}".format(channel, last_event)
    )


def read_mya_point(
    channel: str,
    when: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[float, datetime, float]:
    """
    Read a numeric archived value, falling back to the last valid value at or before 'when'.

    Returns:
        (value, value_timestamp, age_hours)
    """
    reference_time = when or datetime.now()
    value, ts_str, _event = find_last_valid_mya_value(
        channel,
        start_time=reference_time,
        max_lookback_days=max_lookback_days,
    )

    value_time = parse_timestamp(ts_str)
    age_hours = format_age_hours(reference_time, value_time)
    return value, value_time, age_hours


def read_mya_string_point(
    channel: str,
    when: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[str, datetime, float]:
    """
    Read a string archived value, falling back to the last valid value at or before 'when'.

    Returns:
        (value, value_timestamp, age_hours)
    """
    reference_time = when or datetime.now()
    value, ts_str, _event = find_last_valid_mya_string_value(
        channel,
        start_time=reference_time,
        max_lookback_days=max_lookback_days,
    )

    value_time = parse_timestamp(ts_str)
    age_hours = format_age_hours(reference_time, value_time)
    return value, value_time, age_hours


def orientation_index_from_plane_phipol(plane: float, phipol: float) -> int:
    """
    Convert:
      PLANE: PARA=1, PERP=2
      PHIPOL: 0 or 45

    into the orientation index used by the RL environment:
      0: PERP 0/90
      1: PARA 0/90
      2: PERP 45/135
      3: PARA 45/135
    """
    plane_i = int(round(plane))
    phi_i = int(round(phipol))

    if plane_i not in (1, 2):
        raise ValueError("Unexpected HD:CBREM:PLANE value: {0}".format(plane))

    if phi_i not in (0, 45, 135):
        raise ValueError("Unexpected HD:CBREM:PHIPOL value: {0}".format(phipol))

    if plane_i == 2 and phi_i == 0:
        return 0
    if plane_i == 1 and phi_i == 0:
        return 1
    if plane_i == 2 and (phi_i == 45 or phi_i ==135):
        return 2
    if plane_i == 1 and (phi_i == 45 or phi_i == 135):
        return 3

    raise RuntimeError("Unhandled plane/phipol combination")


def read_live_state(*, when: Optional[datetime] = None) -> LiveState:
    beam_energy_E0, beam_energy_time, beam_energy_age_h = read_mya_point(MYA_PVS["beam_energy_E0"], when=when)
    coherent_edge_Ei, target_time, target_age_h = read_mya_point(MYA_PVS["nominal_edge"], when=when)
    peak_energy, peak_time, peak_age_h = read_mya_point(MYA_PVS["measured_edge"], when=when)
    beam_current, current_time, current_age_h = read_mya_point(MYA_PVS["beam_current"], when=when)
    radiator_name, radiator_time, radiator_age_h = read_mya_string_point(MYA_PVS["radiator_name"], when=when)

    # Only require PLANE/PHIPOL to be valid when a diamond is actually in beam.
    if is_diamond_radiator_name(radiator_name):
        plane, plane_time, plane_age_h = read_mya_point(MYA_PVS["plane"], when=when)
        phipol, phipol_time, phipol_age_h = read_mya_point(MYA_PVS["phipol"], when=when)
        orientation_index = orientation_index_from_plane_phipol(plane, phipol)
    else:
        plane = 0.0
        phipol = 0.0
        plane_time = None
        phipol_time = None
        plane_age_h = 0.0
        phipol_age_h = 0.0
        orientation_index = 0  # harmless placeholder; RL is inhibited in this case

    logging.debug(
        "MYA ages at query time %s: beam_E=%.2fh target=%.2fh peak=%.2fh current=%.2fh radiator=%.2fh plane=%.2fh phipol=%.2fh",
        when or datetime.now(),
        beam_energy_age_h,
        target_age_h,
        peak_age_h,
        current_age_h,
        radiator_age_h,
        plane_age_h,
        phipol_age_h,
    )

    return LiveState(
        beam_energy_E0=beam_energy_E0,
        coherent_edge_Ei=coherent_edge_Ei,
        peak_energy=peak_energy,
        dose=0,
        beam_current=beam_current,
        orientation_index=orientation_index,
        radiator_name=radiator_name,
        beam_tilt_pitch_deg=0.0,
        beam_tilt_yaw_deg=0.0,
    )


# ----------------------------------------------------------------------
# Observation builder
# ----------------------------------------------------------------------

def sign_error(peak: float, target: float) -> float:
    return 1.0 if peak > target else -1.0


def build_observation(
    state: LiveState,
    history: ActionHistory,
    *,
    disable_dose_state: bool,
    disable_beam_tilt_state: bool,
) -> np.ndarray:
    """
    Observation format from rl_env.py:
      [beam_E, coh_E, peak, rel_err, dose, ori_idx, sign_err, 5 pitch hist, 5 yaw hist]

    Beam tilt is not part of the uploaded observation, so
    disable_beam_tilt_state is a no-op for the current trained model.
    """
    _ = disable_beam_tilt_state

    relative_error = abs(state.peak_energy - state.coherent_edge_Ei) / (state.coherent_edge_Ei + 1e-8)
    dose_value = 0.0 if disable_dose_state else state.dose

    obs = np.array(
        [
            state.beam_energy_E0 / MAX_ENERGY,
            state.coherent_edge_Ei / MAX_ENERGY,
            state.peak_energy / MAX_ENERGY,
            relative_error,
            dose_value / MAX_DOSE,
            float(state.orientation_index),
            sign_error(state.peak_energy, state.coherent_edge_Ei),
            *history.pitch_padded(),
            *history.yaw_padded(),
        ],
        dtype=np.float32,
    )
    return obs


# ----------------------------------------------------------------------
# Action / geometry helpers
# ----------------------------------------------------------------------

def map_action(a: int) -> int:
    """
    Map {0,1,2} -> {-1,0,+1}, matching rl_env.py.
    """
    return int(a) - 1


def delta_c_from_pitch_yaw(
    delta_h_deg: float,
    delta_v_deg: float,
    phi_deg: float,
    delta_beam_pitch_deg: float = 0.0,
    delta_beam_yaw_deg: float = 0.0,
) -> float:
    """
    Same formula as livingston_sim.py.
    Returns delta_c in radians.
    """
    delta_h_rad = np.deg2rad(delta_h_deg)
    delta_v_rad = np.deg2rad(delta_v_deg)
    delta_beam_h_rad = np.deg2rad(delta_beam_pitch_deg)
    delta_beam_v_rad = np.deg2rad(delta_beam_yaw_deg)
    phi_rad = np.deg2rad(phi_deg)

    delta_h_eff_rad = delta_h_rad + delta_beam_h_rad
    delta_v_eff_rad = delta_v_rad + delta_beam_v_rad

    return delta_v_eff_rad * np.cos(phi_rad) + delta_h_eff_rad * np.sin(phi_rad)


def compute_requests(
    action: np.ndarray,
    orientation_index: int,
    pitch_step_deg: float,
    yaw_step_deg: float,
    *,
    disable_beam_tilt_state: bool,
    beam_tilt_pitch_deg: float = 0.0,
    beam_tilt_yaw_deg: float = 0.0,
) -> Dict[str, float]:
    pitch_a, yaw_a = int(action[0]), int(action[1])

    pitch_dir = map_action(pitch_a)
    yaw_dir = map_action(yaw_a)

    delta_pitch_deg = pitch_dir * pitch_step_deg
    delta_yaw_deg = yaw_dir * yaw_step_deg

    orientation_label = ORIENTATIONS[orientation_index]
    phi_deg = ORIENTATION_TO_PHI[orientation_label]

    if disable_beam_tilt_state:
        beam_tilt_pitch_deg = 0.0
        beam_tilt_yaw_deg = 0.0

    delta_c_rad = delta_c_from_pitch_yaw(
        delta_h_deg=delta_pitch_deg,
        delta_v_deg=delta_yaw_deg,
        phi_deg=phi_deg,
        delta_beam_pitch_deg=beam_tilt_pitch_deg,
        delta_beam_yaw_deg=beam_tilt_yaw_deg,
    )

    return {
        "pitch_dir": float(pitch_dir),
        "yaw_dir": float(yaw_dir),
        "delta_pitch_deg": float(delta_pitch_deg),
        "delta_yaw_deg": float(delta_yaw_deg),
        "delta_c_rad": float(delta_c_rad),
    }


def zero_requests() -> Dict[str, float]:
    return {
        "pitch_dir": 0.0,
        "yaw_dir": 0.0,
        "delta_pitch_deg": 0.0,
        "delta_yaw_deg": 0.0,
        "delta_c_rad": 0.0,
    }


# ----------------------------------------------------------------------
# EPICS write helpers with dry-run support
# ----------------------------------------------------------------------

def write_epics_value(
    pvname: str,
    value,
    *,
    dry_run: bool,
    wait: bool = True,
    timeout: float = 2.0,
) -> None:
    if dry_run:
        logging.info("[DRY RUN] caput %s = %r", pvname, value)
        return

    ok = epics.caput(pvname, value, wait=wait, timeout=timeout)
    if ok != 1:
        raise RuntimeError("caput failed for {0} -> {1}".format(pvname, value))


def write_status(status_code: int, *, dry_run: bool) -> None:
    write_epics_value(WRITE_PVS["status"], status_code, dry_run=dry_run)


def write_heartbeat(counter: int, *, dry_run: bool) -> None:
    write_epics_value(WRITE_PVS["heartbeat"], counter, dry_run=dry_run)


def write_requests(req: Dict[str, float], *, dry_run: bool) -> None:
    write_epics_value(WRITE_PVS["delta_pitch_req"], req["delta_pitch_deg"], dry_run=dry_run)
    write_epics_value(WRITE_PVS["delta_yaw_req"], req["delta_yaw_deg"], dry_run=dry_run)
    write_epics_value(WRITE_PVS["delta_c_req"], req["delta_c_rad"], dry_run=dry_run)


# ----------------------------------------------------------------------
# Replay helpers
# ----------------------------------------------------------------------

def get_query_time(
    *,
    replay_mode: bool,
    replay_time: Optional[datetime],
) -> Optional[datetime]:
    if replay_mode:
        return replay_time
    return None


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def run_loop(
    *,
    model_path: str,
    pitch_step_deg: float,
    yaw_step_deg: float,
    period_s: float,
    dry_run: bool,
    disable_dose_state: bool,
    disable_beam_tilt_state: bool,
    replay_start: Optional[datetime],
    replay_end: Optional[datetime],
    replay_step_s: float,
    min_beam_current: float,
) -> None:
    model = PPO.load(model_path)
    history = ActionHistory()
    heartbeat = 0
    stop_requested = False

    replay_mode = replay_start is not None
    replay_time = replay_start

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logging.info("Loaded model: %s", model_path)
    logging.info("Target PV: %s", MYA_PVS["nominal_edge"])
    logging.info("Measured edge PV: %s", MYA_PVS["measured_edge"])
    logging.info("Radiator PV: %s", MYA_PVS["radiator_name"])
    logging.info("Minimum beam current for AI action: %.3f", min_beam_current)
    
    if dry_run:
        logging.info("Dry-run mode enabled: MYA reads are live/replayed, EPICS writes are suppressed.")
    if disable_dose_state:
        logging.info("Dose state disabled: observation dose term forced to 0.")
    if disable_beam_tilt_state:
        logging.info("Beam-tilt state disabled: no beam-tilt contribution applied.")

    if replay_mode:
        logging.info(
            "Replay mode enabled: start=%s end=%s step=%.3fs",
            replay_start,
            replay_end,
            replay_step_s,
        )

    write_status(STATUS_INIT, dry_run=dry_run)

    try:
        while not stop_requested:
            if replay_mode and replay_end is not None and replay_time is not None and replay_time > replay_end:
                logging.info("Replay finished at %s", replay_time)
                break

            query_time = get_query_time(replay_mode=replay_mode, replay_time=replay_time)
            state = read_live_state(when=query_time)

            diamond_in_beam = is_diamond_radiator_name(state.radiator_name)
            enough_beam_current = state.beam_current >= min_beam_current
            ai_enabled = diamond_in_beam and enough_beam_current
            
            if ai_enabled:
                obs = build_observation(
                    state,
                    history,
                    disable_dose_state=disable_dose_state,
                    disable_beam_tilt_state=disable_beam_tilt_state,
                )

                action, _ = model.predict(obs, deterministic=True)

                req = compute_requests(
                    action=action,
                    orientation_index=state.orientation_index,
                    pitch_step_deg=pitch_step_deg,
                    yaw_step_deg=yaw_step_deg,
                    disable_beam_tilt_state=disable_beam_tilt_state,
                    beam_tilt_pitch_deg=state.beam_tilt_pitch_deg,
                    beam_tilt_yaw_deg=state.beam_tilt_yaw_deg,
                )

                history.append(
                    pitch_dir=int(req["pitch_dir"]),
                    yaw_dir=int(req["yaw_dir"]),
                )

                status_code = STATUS_RUNNING
            else:
                req = zero_requests()
                status_code = STATUS_INHIBITED
                history.clear()

            write_requests(req, dry_run=dry_run)

            heartbeat += 1
            write_heartbeat(heartbeat, dry_run=dry_run)
            write_status(status_code, dry_run=dry_run)

            relative_error = abs(state.peak_energy - state.coherent_edge_Ei) / (state.coherent_edge_Ei + 1e-8)

            prefix = ""
            if replay_mode:
                prefix = "replay_time={0} ".format(query_time)

            logging.info(
                "%sradiator=%s diamond_in_beam=%s target=%.3f peak=%.3f dose=%.3f beam current=%.3f enough_beam_current=%s rel_err=%.6g ori=%s "
                "req_pitch=%+.7f deg req_yaw=%+.7f deg delta_c=%+.9e status=%d",
                prefix,
                state.radiator_name,
                diamond_in_beam,
                state.coherent_edge_Ei,
                state.peak_energy,
                0.0 if disable_dose_state else state.dose,
                state.beam_current,
                enough_beam_current,                
                relative_error,
                ORIENTATIONS[state.orientation_index],
                req["delta_pitch_deg"],
                req["delta_yaw_deg"],
                req["delta_c_rad"],
                status_code,
            )

            if replay_mode:
                replay_time = replay_time + timedelta(seconds=replay_step_s)
                if period_s > 0:
                    time.sleep(period_s)
            else:
                time.sleep(period_s)

    except Exception:
        logging.exception("Bridge failed")
        try:
            write_status(STATUS_ERROR, dry_run=dry_run)
        except Exception:
            logging.exception("Unable to update AI_STATUS to ERROR")
        raise
    finally:
        try:
            write_status(STATUS_STOPPED, dry_run=dry_run)
        except Exception:
            logging.exception("Unable to update AI_STATUS to STOPPED")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GlueX RL -> EPICS bridge using MYA inputs")
    parser.add_argument("--model", required=True, help="Path to trained Stable-Baselines3 PPO model zip")
    parser.add_argument("--pitch-step-deg", type=float, default=2e-4)
    parser.add_argument("--yaw-step-deg", type=float, default=2e-4)
    parser.add_argument("--period-s", type=float, default=1.0, help="Wall-clock loop period in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Read MYA and run the policy, but do not write EPICS PVs")
    parser.add_argument("--disable-dose-state", action="store_true", help="Force the observation dose term to zero")
    parser.add_argument(
        "--disable-beam-tilt-state",
        action="store_true",
        help="Disable any beam-tilt contribution. For the current trained model this is effectively a no-op.",
    )
    parser.add_argument(
        "--replay-start",
        type=str,
        default=None,
        help='Replay archived MYA values starting at "YYYY-MM-DD HH:MM:SS"',
    )
    parser.add_argument(
        "--replay-end",
        type=str,
        default=None,
        help='Stop replay at "YYYY-MM-DD HH:MM:SS"',
    )
    parser.add_argument(
        "--replay-step-s",
        type=float,
        default=1.0,
        help="How much simulated/archive time to advance per replay iteration",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--min-beam-current", type=float, default=100.0, help="Minimum beam current required before the AI is allowed to act, in nA",)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    replay_start = parse_timestamp(args.replay_start) if args.replay_start else None
    replay_end = parse_timestamp(args.replay_end) if args.replay_end else None

    if replay_end is not None and replay_start is None:
        raise ValueError("--replay-end requires --replay-start")

    if replay_start is not None and replay_end is not None and replay_end < replay_start:
        raise ValueError("--replay-end must be >= --replay-start")

    if args.replay_step_s <= 0:
        raise ValueError("--replay-step-s must be > 0")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_loop(
        model_path=args.model,
        pitch_step_deg=args.pitch_step_deg,
        yaw_step_deg=args.yaw_step_deg,
        period_s=args.period_s,
        dry_run=args.dry_run,
        disable_dose_state=args.disable_dose_state,
        disable_beam_tilt_state=args.disable_beam_tilt_state,
        replay_start=replay_start,
        replay_end=replay_end,
        replay_step_s=args.replay_step_s,
        min_beam_current=args.min_beam_current,        
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
