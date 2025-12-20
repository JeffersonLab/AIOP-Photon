#!/usr/bin/env python3
"""
gen_gp_data.py
--------------
Parallel data generator for Gaussian Process surrogate training (FULL physics).
Now includes progress monitoring via tqdm and per-episode reporting.

Run from AIOP-Photon/simulation:
python -m gonio_sim.scripts.gen_gp_data --edge 8.6 --config PARA --phi 0/90 \
  --episodes 60 --steps 200 --nproc 12 --outfile gp_train.jsonl
"""

import argparse, json, random, math, os, time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import numpy as np

from gonio_sim.envs.goniometer_env import GoniometerEnv, EnvConfig


# ============================================================
# Helper functions
# ============================================================

def build_features(env, a_h, a_v, ah_hist, av_hist, run_h, run_v, dir_h, dir_v, E_hist,
                   wobble_pitch_period=0.01, wobble_yaw_period=0.05):
    """Builds the input feature dict from environment + history."""
    yaw_rb = float(env.my_goni.return_set_yaw())
    pitch_rb = float(env.my_goni.return_set_pitch())
    E_ma = float(np.mean(E_hist[-5:])) if E_hist else 0.0
    E_std = float(np.std(E_hist[-5:])) if len(E_hist) > 1 else 0.0

    sin_y = math.sin(2 * math.pi * yaw_rb / wobble_yaw_period)
    cos_y = math.cos(2 * math.pi * yaw_rb / wobble_yaw_period)
    sin_p = math.sin(2 * math.pi * pitch_rb / wobble_pitch_period)
    cos_p = math.cos(2 * math.pi * pitch_rb / wobble_pitch_period)

    return dict(
        yaw_rb=yaw_rb, pitch_rb=pitch_rb,
        a_h=a_h, a_v=a_v,
        dir_h=dir_h, dir_v=dir_v,
        run_h=run_h, run_v=run_v,
        E_ma=E_ma, E_std=E_std,
        sin_y=sin_y, cos_y=cos_y, sin_p=sin_p, cos_p=cos_p,
        ah_hist=list(ah_hist), av_hist=list(av_hist)
    )


def worker_episode(ep_idx, cfg, steps, k_hist, seed, log_interval=50):
    """Simulates one episode with full physics and returns list of feature/target pairs."""
    rng = random.Random(seed + ep_idx)
    env = GoniometerEnv(cfg)  # full physics
    rows = []

    ah, av = [0]*k_hist, [0]*k_hist
    run_h = run_v = 0
    last_dir_h = last_dir_v = 0
    E_hist = []

    t0 = time.time()
    for t in range(steps):
        # Random ±1 action in one axis
        if rng.random() < 0.5:
            a_h, a_v = rng.choice([-1, 0, 1]), 0
        else:
            a_h, a_v = 0, rng.choice([-1, 0, 1])

        dir_h = int(np.sign(a_h))
        dir_v = int(np.sign(a_v))
        run_h = (run_h + 1) if (dir_h != 0 and dir_h == last_dir_h) else (1 if dir_h != 0 else 0)
        run_v = (run_v + 1) if (dir_v != 0 and dir_v == last_dir_v) else (1 if dir_v != 0 else 0)
        last_dir_h = dir_h or last_dir_h
        last_dir_v = dir_v or last_dir_v

        feat = build_features(env, a_h, a_v, ah, av, run_h, run_v, dir_h, dir_v, E_hist)
        E_next = float(env.step(pitch_delta_deg=a_v*0.01, yaw_delta_deg=a_h*0.01))
        E_hist.append(E_next)
        rows.append({"x": feat, "y": E_next})

        ah = (ah + [a_h])[-k_hist:]
        av = (av + [a_v])[-k_hist:]

        # Periodic internal progress update per worker
        if (t + 1) % log_interval == 0 or (t + 1) == steps:
            print(f"[Worker {mp.current_process().name}] Episode {ep_idx} step {t+1}/{steps}")

    print(f"[Worker {mp.current_process().name}] Finished episode {ep_idx} "
          f"in {time.time() - t0:.1f}s, {len(rows)} samples.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Parallel full-physics data generator for GP surrogate.")
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    parser.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--k-hist", type=int, default=5)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--outfile", type=str, default="gp_train.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50,
                        help="print progress every N steps per worker")
    args = parser.parse_args()

    cfg = EnvConfig(edge=args.edge, config=args.config, phi=args.phi)
    n_total = args.episodes * args.steps

    print(f"Generating {n_total} samples "
          f"({args.episodes} episodes × {args.steps} steps) "
          f"using {args.nproc} processes...")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.nproc) as pool, open(args.outfile, "w") as f, tqdm(total=n_total, ncols=100) as pbar:
        job = partial(worker_episode, cfg=cfg, steps=args.steps,
                      k_hist=args.k_hist, seed=args.seed, log_interval=args.log_interval)

        # As episodes complete, update tqdm
        for rows in pool.imap_unordered(job, range(args.episodes)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            pbar.update(len(rows))

    elapsed = time.time() - t0
    print(f"\n✅ Completed {n_total} samples in {elapsed/60:.2f} min "
          f"({elapsed/args.episodes:.1f}s per episode on average)")
    print(f"Output written to {args.outfile}")

if __name__ == "__main__":
    main()
