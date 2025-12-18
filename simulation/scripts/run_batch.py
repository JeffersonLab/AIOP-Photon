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

def run_episode(args):
    """
    Run a full sequential nudge episode inside ONE worker.
    This preserves backlash, wobble, offsets, dose, etc.
    """
    actions, cfg = args

    env = GoniometerEnv(cfg)
    results = []

    for dp, dy in actions:
        result = env.step(dp, dy)
        results.append(result)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parallel stochastic goniometer-beam simulation")
    ap.add_argument("--edge", type=float, required=True)
    ap.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    ap.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    ap.add_argument("--nsteps", type=int, default=200)
    ap.add_argument("--nepisodes", type=int, default=10, help="number of independent episodes")
    ap.add_argument("--nproc", type=int, default=1, help="number of worker processes (cores) to use")
    args = ap.parse_args()

    env_cfg = EnvConfig(edge=args.edge, config=args.config, phi=args.phi)

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
    
    # Generate random step list
    # Change pitch/yaw in degrees
    episodes = []
    for ep_idx in range(args.nepisodes):
        actions = []
        for _ in range(args.nsteps):
            #dp = 1e-3 if random.random() < 0.5 else 0.0
            #dy = 1e-3 if dp == 0.0 else 0.0
            #dp *= 1 if random.random() < 0.5 else -1
            #dy *= 1 if random.random() < 0.5 else -1

            # Randomly choose the sign of the nudge
            sign_dp = 1 if random.random() < 0.5 else -1
            sign_dy = 1 if random.random() < 0.5 else -1
            
            dp = sign_dp * dp_base
            dy = sign_dy * dy_base

            actions.append((dp, dy))

        episodes.append((actions, env_cfg))
            
    print(f"Running {args.nepisodes} episodes × {args.nsteps} steps "
          f"using {args.nproc} processes...")

    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=args.nproc) as pool:
        results_all = list(tqdm(pool.imap(run_episode, episodes),
                            total=args.nepisodes, ncols=90))

    print(f"Completed {args.nepisodes} episodes in {time.time()-t0:.2f}s")

    flattened = [list(row) for episode in results_all for row in episode]    

    # Write output CSV
    outname = f"stochastic_goniometer_beam_{args.config}_{args.phi.replace('/', '-')}.csv"
    #rows = [list(r) for r in results]
    write_csv(outname, rows, HEADER)
    print(f"Saved results to {outname}")
