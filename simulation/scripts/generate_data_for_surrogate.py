#!/usr/bin/env python3
import argparse
import csv
import math
import random
import multiprocessing as mp
from tqdm import tqdm

from gonio_sim.envs.goniometer_env import GoniometerEnv, EnvConfig


# ================================
# Output headers
# ================================

PHYS_HEADER = [
    "yaw_readback", "pitch_readback",
    "yaw_true_nowobble", "pitch_true_nowobble",
    "yaw_true", "pitch_true",
    "beam_delh_eff", "beam_delv_eff",
    "thetah_eff", "thetav_eff",
    "beamx", "beamy", "peak_energy",
    "Ga_deg", "Gv_deg", "Gh_deg"
]

EXTRA_HEADER = [
    "episode", "step",
    "dp", "dy",
    "dir_pitch", "dir_yaw",
    "run_pitch", "run_yaw"
]


# ================================
# Worker function
# ================================
def run_episode_worker(args):
    """
    Worker wrapper to run one episode sequentially.
    args = (episode_index, nsteps, dp_base, dy_base, cfg)
    """
    ep_idx, nsteps, dp_base, dy_base, cfg = args

    env = GoniometerEnv(cfg)

    rows = []
    run_pitch = 0
    run_yaw = 0
    last_dir_pitch = 0
    last_dir_yaw = 0

    for step in range(nsteps):

        # Random sign choices
        sign_dp = 1 if random.random() < 0.5 else -1
        sign_dy = 1 if random.random() < 0.5 else -1

        dp = sign_dp * dp_base
        dy = sign_dy * dy_base

        # Direction indicators
        dir_pitch = 0 if dp == 0 else (1 if dp > 0 else -1)
        dir_yaw   = 0 if dy == 0 else (1 if dy > 0 else -1)

        # Run-length counters
        if dir_pitch != 0 and dir_pitch == last_dir_pitch:
            run_pitch += 1
        elif dir_pitch != 0:
            run_pitch = 1
        else:
            run_pitch = 0

        if dir_yaw != 0 and dir_yaw == last_dir_yaw:
            run_yaw += 1
        elif dir_yaw != 0:
            run_yaw = 1
        else:
            run_yaw = 0

        last_dir_pitch = dir_pitch if dir_pitch != 0 else last_dir_pitch
        last_dir_yaw   = dir_yaw   if dir_yaw   != 0 else last_dir_yaw

        # Run physics
        phys = env.step(dp, dy)

        # Assemble row
        row = {
            "episode": ep_idx,
            "step": step,
            "dp": dp,
            "dy": dy,
            "dir_pitch": dir_pitch,
            "dir_yaw": dir_yaw,
            "run_pitch": run_pitch,
            "run_yaw": run_yaw,
        }

        for k, v in zip(PHYS_HEADER, phys):
            row[k] = v

        rows.append(row)

    return rows


# ================================
# Save CSV
# ================================
def write_csv(filename, rows):
    header = EXTRA_HEADER + PHYS_HEADER
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# ================================
# Main
# ================================
def main():

    parser = argparse.ArgumentParser(description="Generate XGBoost training data (multicore)")
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    parser.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--outfile", type=str, default="xgb_train.csv")
    args = parser.parse_args()

    cfg = EnvConfig(edge=args.edge, config=args.config, phi=args.phi)

    # ------------------------------
    # Get φ (crystal azimuth)
    # ------------------------------
    phi_map = {
        ("PARA", "0/90"):   0,
        ("PERP", "0/90"):  90,
        ("PARA", "45/135"): 135,
        ("PERP", "45/135"): 45,
    }

    phi_deg = phi_map[(args.config, args.phi)]
    phi_rad = math.radians(phi_deg)

    dp_base = 1e-3 * math.cos(phi_rad)
    dy_base = 1e-3 * math.sin(phi_rad)

    # ------------------------------
    # Create worker input list
    # ------------------------------
    tasks = [
        (ep_idx, args.steps, dp_base, dy_base, cfg)
        for ep_idx in range(args.episodes)
    ]

    print(f"Generating {args.episodes} episodes × {args.steps} steps using {args.nproc} cores...")

    # ------------------------------
    # Run workers
    # ------------------------------
    all_rows = []
    with mp.get_context("spawn").Pool(processes=args.nproc) as pool:
        for episode_rows in tqdm(pool.imap(run_episode_worker, tasks), total=args.episodes, ncols=80):
            all_rows.extend(episode_rows)

    # ------------------------------
    # Save results
    # ------------------------------
    print(f"Saving {len(all_rows)} rows → {args.outfile}")
    write_csv(args.outfile, all_rows)
    print("Done.")


if __name__ == "__main__":
    main()
 
