#!/usr/bin/env python3
"""
GlueX RL -> EPICS bridge.

Reads live/archived MYA values, builds the observation expected by the trained
Stable-Baselines3 PPO policy, and writes delta-pitch / delta-yaw / delta-c
requests back to EPICS.

IMPORTANT: the observation and action encoding here MUST match
rl_env_moreAction.CoherentGoniometerEnv exactly. This file was updated to the
current single-Discrete-action, 12-dim observation environment:

    Observation (12-dim):
        [beam_E/MAX_E, coh_E/MAX_E, peak/MAX_E, rel_err, dose/MAX_DOSE,
         ori_index(raw), sign_err, *5 normalized action-history]
    Action: Discrete(2*M+1) mapped to [-M, +M] via a - M, where
        M = max_step_multiplier. A single action drives pitch/yaw per orientation.

Note on backlash: the physical goniometer motors have mechanical backlash
(~2.1 mdeg pitch, ~4.1 mdeg yaw), which the training simulator now models. This
bridge only emits incremental requests; the real motors provide the actual
backlash. The pitch/yaw setpoint-vs-readback PVs are logged so engagement can be
monitored offline.
"""
from __future__ import annotations

import argparse
import logging
import re
import os
import signal
import sys
import time
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
ACTION_HISTORY_LENGTH = 5
# Must match the env's err_scale_mev used at training time (obs[3] scaling).
ERR_SCALE_MEV = 15.0

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
    "phi022": "HD:CBREM:PHI022",
    "pitch_setpoint": "HD:GONI:PITCH",
    "pitch_readback": "HD:GONI:PITCH.RBV",
    "yaw_setpoint": "HD:GONI:YAW",
    "yaw_readback": "HD:GONI:YAW.RBV",
    "radiator_name": "HD:GONI:RADIATOR_NAME",
    "fit_chi2": "HD:CBREM:FIT_CHI2",
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
    # required
    beam_energy_E0: float
    coherent_edge_Ei: float
    peak_energy: float
    dose: float
    beam_current: float
    orientation_index: int
    radiator_name: str
    # optional (must follow the required fields for a valid dataclass)
    pitch_setpoint: float = 0.0
    pitch_readback: float = 0.0
    yaw_setpoint: float = 0.0
    yaw_readback: float = 0.0
    beam_tilt_pitch_deg: float = 0.0
    beam_tilt_yaw_deg: float = 0.0
    fit_chi2: Optional[float] = None  # None => FIT_CHI2 PV unavailable (e.g. old data)


class ActionHistory:
    """
    Matches rl_env_moreAction.CoherentGoniometerEnv:
      - a SINGLE combined action history (not separate pitch/yaw)
      - only nonzero mapped action directions are stored
      - only the last ACTION_HISTORY_LENGTH are kept
      - stored as raw directions in [-M, +M]; normalized at observation time
    """

    def __init__(self, length: int = ACTION_HISTORY_LENGTH) -> None:
        self.length = length
        self.actions: deque = deque(maxlen=length)

    def append(self, action_dir: int) -> None:
        if action_dir != 0:
            self.actions.append(int(action_dir))

    def clear(self) -> None:
        self.actions.clear()

    def padded(self) -> List[float]:
        pad = [0.0] * (self.length - len(self.actions))
        return pad + [float(a) for a in self.actions]


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
    query = PointQuery(channel=channel, time=when or datetime.now())
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
    query = PointQuery(channel=channel, time=when or datetime.now())
    point = Point(query)
    point.run()
    event = point.event or {}
    data = event.get("data", {})
    if "v" in data:
        return str(data["v"]).strip(), data.get("d"), event
    return None, data.get("d"), event


def _candidate_times(base_time: datetime, max_lookback_days: int) -> List[datetime]:
    return [
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


def find_last_valid_mya_value(
    channel: str,
    start_time: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[float, str, dict]:
    base_time = start_time or datetime.now()
    last_event = None
    for t in _candidate_times(base_time, max_lookback_days):
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
    base_time = start_time or datetime.now()
    last_event = None
    for t in _candidate_times(base_time, max_lookback_days):
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
    reference_time = when or datetime.now()
    value, ts_str, _event = find_last_valid_mya_value(
        channel, start_time=reference_time, max_lookback_days=max_lookback_days
    )
    value_time = parse_timestamp(ts_str)
    return value, value_time, format_age_hours(reference_time, value_time)


def read_mya_string_point(
    channel: str,
    when: Optional[datetime] = None,
    max_lookback_days: int = 30,
) -> Tuple[str, datetime, float]:
    reference_time = when or datetime.now()
    value, ts_str, _event = find_last_valid_mya_string_value(
        channel, start_time=reference_time, max_lookback_days=max_lookback_days
    )
    value_time = parse_timestamp(ts_str)
    return value, value_time, format_age_hours(reference_time, value_time)


def orientation_index_from_plane_phipol(plane: float, phipol: float) -> int:
    """
    PLANE: PARA=1, PERP=2 ; PHIPOL: 0 or 45  ->  RL orientation index:
      0: PERP 0/90, 1: PARA 0/90, 2: PERP 45/135, 3: PARA 45/135
    """
    plane_i = int(round(plane))
    phi_i = int(round(phipol))

    if plane_i not in (1, 2):
        raise ValueError("Unexpected HD:CBREM:PLANE value: {0}".format(plane))
    if phi_i not in (0, 45, 90, 135, 180):
        raise ValueError("Unexpected HD:CBREM:PHIPOL value: {0}".format(phipol))

    if phi_i in (90, 180):
        phi_i = 0

    if plane_i == 2 and phi_i == 0:
        return 0
    if plane_i == 1 and phi_i == 0:
        return 1
    if plane_i == 2 and phi_i in (45, 135):
        return 2
    if plane_i == 1 and phi_i in (45, 135):
        return 3
    raise RuntimeError("Unhandled plane/phipol combination")


def calculate_phipol_from_plane_phi022(plane: float, phi022: float) -> float:
    plane_i = int(round(plane))
    if plane_i == 1:
        return 180.0 - phi022
    if plane_i == 2:
        return 90.0 - phi022
    raise ValueError("Unexpected HD:CBREM:PLANE value: {0}".format(plane))


def read_live_state(*, when: Optional[datetime] = None) -> LiveState:
    beam_energy_E0, _, beam_energy_age_h = read_mya_point(MYA_PVS["beam_energy_E0"], when=when)
    coherent_edge_Ei, _, target_age_h = read_mya_point(MYA_PVS["nominal_edge"], when=when)
    peak_energy, _, peak_age_h = read_mya_point(MYA_PVS["measured_edge"], when=when)
    beam_current, _, current_age_h = read_mya_point(MYA_PVS["beam_current"], when=when)
    radiator_name, _, radiator_age_h = read_mya_string_point(MYA_PVS["radiator_name"], when=when)
    pitch_setpoint, _, _ = read_mya_point(MYA_PVS["pitch_setpoint"], when=when)
    pitch_readback, _, _ = read_mya_point(MYA_PVS["pitch_readback"], when=when)
    yaw_setpoint, _, _ = read_mya_point(MYA_PVS["yaw_setpoint"], when=when)
    yaw_readback, _, _ = read_mya_point(MYA_PVS["yaw_readback"], when=when)
    # FIT_CHI2 was only added recently, so it is absent from older MYA data.
    # Treat a missing PV as "unavailable" (None) instead of failing the whole read.
    try:
        fit_chi2, _, _ = read_mya_point(MYA_PVS["fit_chi2"], when=when)
    except RuntimeError:
        fit_chi2 = None
        logging.debug("FIT_CHI2 unavailable at %s; chi2 gate will be skipped",
                      when or datetime.now())

    # Only require PLANE/PHIPOL to be valid when a diamond is actually in beam.
    if is_diamond_radiator_name(radiator_name):
        plane, _, plane_age_h = read_mya_point(MYA_PVS["plane"], when=when)
        try:
            phipol, _, phipol_age_h = read_mya_point(MYA_PVS["phipol"], when=when)
        except RuntimeError:
            phi022, _, phipol_age_h = read_mya_point(MYA_PVS["phi022"], when=when)
            phipol = calculate_phipol_from_plane_phi022(plane, phi022)
        orientation_index = orientation_index_from_plane_phipol(plane, phipol)
    else:
        plane_age_h = 0.0
        phipol_age_h = 0.0
        orientation_index = 0  # harmless placeholder; RL is inhibited in this case

    logging.debug(
        "MYA ages at %s: beam_E=%.2fh target=%.2fh peak=%.2fh current=%.2fh "
        "radiator=%.2fh plane=%.2fh phipol=%.2fh",
        when or datetime.now(),
        beam_energy_age_h, target_age_h, peak_age_h, current_age_h,
        radiator_age_h, plane_age_h, phipol_age_h,
    )

    return LiveState(
        beam_energy_E0=beam_energy_E0,
        coherent_edge_Ei=coherent_edge_Ei,
        peak_energy=peak_energy,
        dose=0.0,
        beam_current=beam_current,
        orientation_index=orientation_index,
        radiator_name=radiator_name,
        pitch_setpoint=pitch_setpoint,
        pitch_readback=pitch_readback,
        yaw_setpoint=yaw_setpoint,
        yaw_readback=yaw_readback,
        beam_tilt_pitch_deg=0.0,
        beam_tilt_yaw_deg=0.0,
        fit_chi2=fit_chi2,
    )


# ----------------------------------------------------------------------
# Observation builder  (12-dim, must match the env exactly)
# ----------------------------------------------------------------------

def sign_error(peak: float, target: float) -> float:
    return 1.0 if peak > target else -1.0


def build_observation(
    state: LiveState,
    history: ActionHistory,
    *,
    max_step_multiplier: int,
    disable_dose_state: bool,
) -> np.ndarray:
    # obs[3] must match the env: signed error scaled to O(1), clipped to [-1, 1].
    signed_scaled_err = float(
        np.clip((state.peak_energy - state.coherent_edge_Ei) / ERR_SCALE_MEV, -1.0, 1.0)
    )
    dose_value = 0.0 if disable_dose_state else state.dose
    norm_history = [a / max_step_multiplier for a in history.padded()]

    obs = np.array(
        [
            state.beam_energy_E0 / MAX_ENERGY,
            state.coherent_edge_Ei / MAX_ENERGY,
            state.peak_energy / MAX_ENERGY,
            signed_scaled_err,
            dose_value / MAX_DOSE,
            float(state.orientation_index),
            sign_error(state.peak_energy, state.coherent_edge_Ei),
            *norm_history,
        ],
        dtype=np.float32,
    )
    return obs


# ----------------------------------------------------------------------
# Action / geometry helpers
# ----------------------------------------------------------------------

def map_action(a: int, max_step_multiplier: int) -> int:
    """Map Discrete index 0..2M -> signed multiplier -M..+M (matches env)."""
    return int(a) - max_step_multiplier


def delta_c_from_pitch_yaw(
    delta_h_deg: float,
    delta_v_deg: float,
    phi_deg: float,
    delta_beam_pitch_deg: float = 0.0,
    delta_beam_yaw_deg: float = 0.0,
) -> float:
    delta_h_rad = np.deg2rad(delta_h_deg)
    delta_v_rad = np.deg2rad(delta_v_deg)
    delta_beam_h_rad = np.deg2rad(delta_beam_pitch_deg)
    delta_beam_v_rad = np.deg2rad(delta_beam_yaw_deg)
    phi_rad = np.deg2rad(phi_deg)
    delta_h_eff_rad = delta_h_rad + delta_beam_h_rad
    delta_v_eff_rad = delta_v_rad + delta_beam_v_rad
    return delta_v_eff_rad * np.cos(phi_rad) + delta_h_eff_rad * np.sin(phi_rad)


def compute_requests(
    action,
    orientation_index: int,
    pitch_step_deg: float,
    yaw_step_deg: float,
    max_step_multiplier: int,
    *,
    disable_beam_tilt_state: bool,
    beam_tilt_pitch_deg: float = 0.0,
    beam_tilt_yaw_deg: float = 0.0,
) -> Dict[str, float]:
    # model.predict returns a scalar (0-d array) for a Discrete action space
    action_dir = map_action(int(np.asarray(action).item()), max_step_multiplier)

    if orientation_index == 0:      # PERP 0/90
        pitch_dir, yaw_dir = action_dir, 0
    elif orientation_index == 1:    # PARA 0/90
        pitch_dir, yaw_dir = 0, action_dir
    elif orientation_index == 2:    # PERP 45/135
        pitch_dir, yaw_dir = -action_dir, action_dir
    elif orientation_index == 3:    # PARA 45/135
        pitch_dir, yaw_dir = action_dir, action_dir
    else:
        raise ValueError("Invalid orientation index: {0}".format(orientation_index))

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
        "action_dir": float(action_dir),
        "pitch_dir": float(pitch_dir),
        "yaw_dir": float(yaw_dir),
        "delta_pitch_deg": float(delta_pitch_deg),
        "delta_yaw_deg": float(delta_yaw_deg),
        "delta_c_rad": float(delta_c_rad),
    }


def zero_requests() -> Dict[str, float]:
    return {
        "action_dir": 0.0,
        "pitch_dir": 0.0,
        "yaw_dir": 0.0,
        "delta_pitch_deg": 0.0,
        "delta_yaw_deg": 0.0,
        "delta_c_rad": 0.0,
    }


# ----------------------------------------------------------------------
# EPICS write helpers with dry-run support
# ----------------------------------------------------------------------

def write_epics_value(pvname: str, value, *, dry_run: bool, wait: bool = True, timeout: float = 2.0) -> None:
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

def get_query_time(*, replay_mode: bool, replay_time: Optional[datetime]) -> Optional[datetime]:
    return replay_time if replay_mode else None


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def run_loop(
    *,
    model_path: str,
    pitch_step_deg: float,
    yaw_step_deg: float,
    max_step_multiplier: int,
    period_s: float,
    dry_run: bool,
    disable_dose_state: bool,
    disable_beam_tilt_state: bool,
    replay_start: Optional[datetime],
    replay_end: Optional[datetime],
    replay_step_s: float,
    min_beam_current: float,
    max_fit_chi2: float,
) -> None:
    model = PPO.load(model_path)

    # Fail fast if the model's observation/action spaces do not match this bridge.
    expected_obs = 7 + ACTION_HISTORY_LENGTH
    obs_dim = int(np.prod(model.observation_space.shape))
    if obs_dim != expected_obs:
        raise RuntimeError(
            "Model observation dim {0} != bridge observation dim {1}. "
            "The bridge and the training env are out of sync.".format(obs_dim, expected_obs)
        )
    n_actions = getattr(model.action_space, "n", None)
    if n_actions is not None and n_actions != (2 * max_step_multiplier + 1):
        raise RuntimeError(
            "Model has {0} actions but --max-step-multiplier={1} implies {2}. "
            "Pass the same max_step_multiplier used in training.".format(
                n_actions, max_step_multiplier, 2 * max_step_multiplier + 1
            )
        )

    history = ActionHistory(length=ACTION_HISTORY_LENGTH)
    heartbeat = 0
    stop_requested = False

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = os.path.join(log_dir, f"goni_rl_backend_{timestamp}.csv")

    csv_log_file = open(csv_log_path, "a", newline="")
    csv_writer = csv.writer(csv_log_file)
    csv_writer.writerow([
        "timestamp", "radiator", "diamond_in_beam", "target", "peak",
        "fit_chi2", "good_fit", "dose", "beam_current", "enough_beam_current",
        "relative_error", "orientation",
        "pitch_setpoint", "pitch_readback", "yaw_setpoint", "yaw_readback",
        "action_dir", "req_pitch_deg", "req_yaw_deg", "delta_c_rad", "status",
    ])
    csv_log_file.flush()

    replay_mode = replay_start is not None
    replay_time = replay_start

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logging.info("Loaded model: %s (obs_dim=%d, n_actions=%s)", model_path, obs_dim, n_actions)
    logging.info("Target PV: %s | Measured edge PV: %s | Radiator PV: %s",
                 MYA_PVS["nominal_edge"], MYA_PVS["measured_edge"], MYA_PVS["radiator_name"])
    logging.info("Minimum beam current for AI action: %.3f nA", min_beam_current)
    if dry_run:
        logging.info("Dry-run: MYA reads live/replayed, EPICS writes suppressed.")
    if disable_dose_state:
        logging.info("Dose state disabled: observation dose term forced to 0.")
    if replay_mode:
        logging.info("Replay: start=%s end=%s step=%.3fs", replay_start, replay_end, replay_step_s)

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
            # Only gate on fit quality when the FIT_CHI2 PV is available. On older
            # data the PV does not exist (state.fit_chi2 is None); in that case we
            # do not block AI action on a check we cannot perform.
            if state.fit_chi2 is None:
                good_fit = True
            else:
                good_fit = state.fit_chi2 <= max_fit_chi2

            ai_enabled = diamond_in_beam and enough_beam_current and good_fit

            if ai_enabled:
                obs = build_observation(
                    state, history,
                    max_step_multiplier=max_step_multiplier,
                    disable_dose_state=disable_dose_state,
                )
                action, _ = model.predict(obs, deterministic=True)
                req = compute_requests(
                    action=action,
                    orientation_index=state.orientation_index,
                    pitch_step_deg=pitch_step_deg,
                    yaw_step_deg=yaw_step_deg,
                    max_step_multiplier=max_step_multiplier,
                    disable_beam_tilt_state=disable_beam_tilt_state,
                    beam_tilt_pitch_deg=state.beam_tilt_pitch_deg,
                    beam_tilt_yaw_deg=state.beam_tilt_yaw_deg,
                )
                history.append(int(req["action_dir"]))
                status_code = STATUS_RUNNING
            else:
                req = zero_requests()
                status_code = STATUS_INHIBITED
                history.clear()

            write_requests(req, dry_run=dry_run)
            heartbeat += 1
            write_heartbeat(heartbeat, dry_run=dry_run)
            write_status(status_code, dry_run=dry_run)

            chi2_str = "n/a" if state.fit_chi2 is None else "{0:.4f}".format(state.fit_chi2)
            relative_error = abs(state.peak_energy - state.coherent_edge_Ei) / (state.coherent_edge_Ei + 1e-8)
            prefix = "replay_time={0} ".format(query_time) if replay_mode else ""
            logging.info(
                "%sradiator=%s diamond_in_beam=%s target=%.3f peak=%.3f fit_chi2=%s good_fit=%s "
                "dose=%.3f beam_current=%.3f enough=%s rel_err=%.6g ori=%s "
                "action=%+d req_pitch=%+.7f req_yaw=%+.7f delta_c=%+.9e status=%d",
                prefix, state.radiator_name, diamond_in_beam, state.coherent_edge_Ei,
                state.peak_energy, chi2_str, good_fit,
                0.0 if disable_dose_state else state.dose, state.beam_current, enough_beam_current,
                relative_error, ORIENTATIONS[state.orientation_index],
                int(req["action_dir"]), req["delta_pitch_deg"], req["delta_yaw_deg"],
                req["delta_c_rad"], status_code,
            )

            csv_writer.writerow([
                (query_time or datetime.now()).isoformat(),
                state.radiator_name, diamond_in_beam,
                state.coherent_edge_Ei, state.peak_energy,
                state.fit_chi2, good_fit,
                0.0 if disable_dose_state else state.dose,
                state.beam_current, enough_beam_current,
                relative_error, ORIENTATIONS[state.orientation_index],
                f"{state.pitch_setpoint:.6f}", f"{state.pitch_readback:.6f}",
                f"{state.yaw_setpoint:.6f}", f"{state.yaw_readback:.6f}",
                int(req["action_dir"]), req["delta_pitch_deg"], req["delta_yaw_deg"],
                req["delta_c_rad"], status_code,
            ])
            csv_log_file.flush()

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
        try:
            csv_log_file.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GlueX RL -> EPICS bridge using MYA inputs")
    parser.add_argument("--model", required=True, help="Path to trained SB3 PPO model zip")
    parser.add_argument("--pitch-step-deg", type=float, default=2e-4)
    parser.add_argument("--yaw-step-deg", type=float, default=2e-4)
    parser.add_argument("--max-step-multiplier", type=int, default=10,
                        help="Must match max_step_multiplier used in training")
    parser.add_argument("--period-s", type=float, default=1.0, help="Wall-clock loop period in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Run the policy but do not write EPICS PVs")
    parser.add_argument("--disable-dose-state", action="store_true", help="Force the observation dose term to zero")
    parser.add_argument("--disable-beam-tilt-state", action="store_true",
                        help="Disable beam-tilt contribution (no-op for the current observation).")
    parser.add_argument("--replay-start", type=str, default=None,
                        help='Replay archived MYA values starting at "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--replay-end", type=str, default=None, help='Stop replay at "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--replay-step-s", type=float, default=1.0,
                        help="How much archive time to advance per replay iteration")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--min-beam-current", type=float, default=100.0,
                        help="Minimum beam current (nA) required before the AI is allowed to act")
    parser.add_argument("--max-fit-chi2", type=float, default=5.0,
                        help="Maximum allowed fit chi2 before AI is inhibited")
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
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_loop(
        model_path=args.model,
        pitch_step_deg=args.pitch_step_deg,
        yaw_step_deg=args.yaw_step_deg,
        max_step_multiplier=args.max_step_multiplier,
        period_s=args.period_s,
        dry_run=args.dry_run,
        disable_dose_state=args.disable_dose_state,
        disable_beam_tilt_state=args.disable_beam_tilt_state,
        replay_start=replay_start,
        replay_end=replay_end,
        replay_step_s=args.replay_step_s,
        min_beam_current=args.min_beam_current,
        max_fit_chi2=args.max_fit_chi2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
