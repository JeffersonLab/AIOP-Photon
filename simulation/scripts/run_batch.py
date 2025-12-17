#!/usr/bin/env python3
"""
run_batch.py
------------
Multiprocess batch simulator for goniometer + beam system.
This script reproduces the stochastic nudge simulation from the original
sim_wrapper_with_backlash_and_wobble.py but using the modular GoniometerEnv.
"""

import argparse
import time
import random
import multiprocessing as mp
from tqdm import tqdm

from gonio_sim.envs.goniometer_env import GoniometerEnv, EnvConfig
from gonio_sim.utils.io import write_csv

HEADER = [
    "yaw_readback", "pitch_readback",
    "yaw_true_nowobble", "pitch_true_nowobble",
    "yaw_true", "pitch_true",
    "beam_delh_eff", "beam_delv_eff",
    "thetah_eff", "thetav_eff",
    "beamx", "beamy", "peak_energy",
    "Ga_deg", "Gv_deg", "Gh_deg"
]


def run_single_step(args_tuple):
    """Helper function for multiprocessing pool."""
    dp, dy, env_cfg = args_tuple
    # Each worker creates its own environment (ROOT not thread-safe)
    env = GoniometerEnv(env_cfg)
    result = env.step(dp, dy)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parallel stochastic goniometer-beam simulation")
    ap.add_argument("--edge", type=float, required=True)
    ap.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    ap.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    ap.add_argument("--nsteps", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=1, help="number of worker processes (cores) to use")
    args = ap.parse_args()

    env_cfg = EnvConfig(edge=args.edge, config=args.config, phi=args.phi)

    # Generate random step list (like original sim_wrapper)
    actions = []
    for _ in range(args.nsteps):
        dp = 0.01 if random.random() < 0.5 else 0.0
        dy = 0.01 if dp == 0.0 else 0.0
        dp *= 1 if random.random() < 0.5 else -1
        dy *= 1 if random.random() < 0.5 else -1
        actions.append((dp, dy, env_cfg))

    print(f"Running {args.nsteps} steps using {args.nproc} processes...")

    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=args.nproc) as pool:
        results = list(tqdm(pool.imap(run_single_step, actions),
                            total=len(actions), ncols=90))

    print(f"Completed {len(results)} steps in {time.time()-t0:.2f}s total")

    # Write output CSV
    outname = f"stochastic_goniometer_beam_{args.config}_{args.phi.replace('/', '-')}.csv"
    rows = [list(r) for r in results]
    write_csv(outname, rows, HEADER)
    print(f"Saved results to {outname}")
