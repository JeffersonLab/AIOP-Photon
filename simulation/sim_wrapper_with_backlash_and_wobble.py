#!/usr/bin/env python3
import os
import csv
import time
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass
from damage_model import DiamondDamageModel
from spotfinder import make_beamtilt, snap_crystal_orientation

# ==============================================================
#  AxisBacklashState and backlash-aware stepping
# ==============================================================

@dataclass
class AxisBacklashState:
    last_dir: int = 0
    skip_left: float = 0.0
    pos_true: float = 0.0
    pos_readback: float = 0.0


def step_with_backlash(target, step, backlash_n, state):
    delta = target - state.pos_true
    if abs(delta) < 1e-12:
        state.pos_readback = target
        return state.pos_true, state.pos_readback

    new_dir = 1 if delta > 0 else -1

    # Detect direction change
    if new_dir != state.last_dir:
        state.last_dir = new_dir
        state.skip_left = float(backlash_n)

    # Controller readback always moves
    move_readback = new_dir * min(abs(delta), step)
    state.pos_readback += move_readback

    # Apply fractional backlash consumption
    if state.skip_left > 0.0:
        if state.skip_left >= 1.0:
            state.skip_left -= 1.0
            move_true = 0.0
        else:
            move_true = (1.0 - state.skip_left) * move_readback
            state.skip_left = 0.0
    else:
        move_true = move_readback

    state.pos_true += move_true
    return state.pos_true, state.pos_readback


# ==============================================================
#  Goniometer wobble model (fixed from Yi study)
# ==============================================================

def apply_wobble(angle, amplitude, period, phase):
    if amplitude == 0.0 or period == 0.0:
        return angle
    return angle + amplitude * np.sin(2 * np.pi * angle / period + phase)


# ==============================================================
#  Goniometer + ROOT simulation logic
# ==============================================================

def compute_goniometer_angles(phi, c, i, offsets):
    phi0, Bv, Bh, Theta, Phi = (
        offsets["phi0"], offsets["Bv"], offsets["Bh"], offsets["Theta"], offsets["Phi"]
    )
    Ga = phi - phi0
    Gv = c*np.cos(phi) - i*np.sin(phi) - Theta*np.cos(Ga + Phi) + Bv
    Gh = c*np.sin(phi) + i*np.cos(phi) - Theta*np.sin(Ga + Phi) + Bh
    return Ga, Gv, Gh


def init_root(base_args):
    import ROOT
    libdir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gSystem.AddDynamicPath(libdir)
    ROOT.gSystem.Load(os.path.join(libdir, "CobremsGeneration_cc.so"))
    #ROOT.gSystem.Load(os.path.join(libdir, "rootvisuals_C.so"))

    ebeam, ibeam = base_args["ebeam"], base_args["ibeam"]
    ROOT.cobrems = ROOT.CobremsGeneration(ebeam, ibeam)
    ROOT.cobrems.setBeamErms(base_args["ebeamrms"])
    ROOT.cobrems.setBeamEmittance(base_args["emittance"])
    ROOT.cobrems.setCollimatorSpotrms(base_args["vspotrms"] * 1e-3)
    ROOT.cobrems.setCollimatorDistance(base_args["coldist"])
    ROOT.cobrems.setCollimatorDiameter(base_args["coldiam"] * 1e-3)
    ROOT.cobrems.setTargetCrystal("diamond")
    ROOT.cobrems.setTargetThickness(base_args["radthick"] * 1e-6)
    return ROOT


def compute_peak_energy(params):
    (i, total, yaw_true, pitch_true, yaw_readback, pitch_readback,
     beam_delh, beam_delv, base_args, dose, damage, offsets) = params

    ROOT = init_root(base_args)

    h_incoh = ROOT.amorph_intensity(
        base_args["ebeam"], base_args["ibeam"],
        base_args["peresol"], base_args["penergy0"], base_args["penergy1"]
    )[0]
    h_incoh.SetDirectory(0)

    # Apply random beam tilt
    beam_delh_eff = beam_delh
    beam_delv_eff = beam_delv

    thetah_eff = yaw_true + beam_delh_eff
    thetav_eff = pitch_true + beam_delv_eff
    base_args["thetah"], base_args["thetav"] = thetah_eff, thetav_eff

    beamx = np.random.normal(base_args["xoffset"], base_args["beamx_noise"])
    beamy = np.random.normal(base_args["yoffset"], base_args["beamy_noise"])
    base_args["xoffset"], base_args["yoffset"] = beamx, beamy

    hlist = ROOT.cobrems_intensity(
        base_args["radname"], base_args["iradview"],
        base_args["ebeam"], base_args["ibeam"],
        base_args["xyresol"], base_args["thetah"], base_args["thetav"],
        base_args["xoffset"], base_args["yoffset"], base_args["phideg"],
        base_args["xsigma"], base_args["ysigma"], base_args["xycorr"],
        base_args["peresol"], base_args["penergy0"], base_args["penergy1"], 0
    )
    h_coh = hlist[0]
    h_coh.SetDirectory(0)

    n_bins = h_coh.GetNbinsX()
    energy = np.array([h_coh.GetBinCenter(b) for b in range(1, n_bins + 1)])
    y_coh = np.array([h_coh.GetBinContent(b) for b in range(1, n_bins + 1)])
    y_inc = np.array([h_incoh.GetBinContent(b) for b in range(1, n_bins + 1)])

    E_MeV = 1000.0 * energy
    y_coh = damage.apply(E_MeV, y_coh, dose)
    enhancement = np.divide(y_coh, y_inc, out=np.zeros_like(y_coh), where=y_inc != 0)
    peak_energy = float(energy[np.nanargmax(enhancement)])

    phi = np.deg2rad(base_args["phideg"])
    c, i_angle = pitch_true * 1e-3, yaw_true * 1e-3
    Ga, Gv, Gh = compute_goniometer_angles(phi, c, i_angle, offsets)

    return (yaw_readback, pitch_readback, yaw_true, pitch_true,
            beam_delh_eff, beam_delv_eff,
            thetah_eff, thetav_eff,
            beamx, beamy, peak_energy,
            np.rad2deg(Ga), np.rad2deg(Gv), np.rad2deg(Gh))


# ==============================================================
#  Main driver (stochastic motion)
# ==============================================================

def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Stochastic coherent bremsstrahlung simulator (goniometer + beam)")
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    parser.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    parser.add_argument("--dose", type=float, default=0.0)
    parser.add_argument("--nproc", type=int, default=12)
    parser.add_argument("--backlash-n", type=float, default=2.0, help="number of backlash steps (can be fractional)")
    parser.add_argument("--step", type=float, default=0.01745, help="nudge step size [mrad] (1 millidegree)")
    parser.add_argument("--nsteps", type=int, default=1000, help="number of stochastic nudges to simulate")
    args = parser.parse_args()

    ebeam, ibeam = 11.7, 2.2
    base_args = {
        "radname": "JD70-103", "iradview": 0,
        "ebeam": ebeam, "ibeam": ibeam,
        "xyresol": 0.01, "xoffset": -0.38, "yoffset": 3.13,
        "xsigma": 1.0, "ysigma": 0.5, "xycorr": 0.42,
        "peresol": 0.02, "penergy0": 4.5, "penergy1": 11.0,
        "tiltrange": 1.0, "tiltresol": 0.01,
        "ebeamrms": 0.001, "emittance": 4.2e-9,
        "vspotrms": 0.5, "coldiam": 3.4, "coldist": 76.0,
        "radthick": 50.0, "snapedge": args.edge,
        "beamx_noise": 0.1, "beamy_noise": 0.1,
    }

    phi_map = {("PARA", "0/90"): 0, ("PERP", "0/90"): 90,
               ("PARA", "45/135"): 135, ("PERP", "45/135"): 45}
    pol_dir = args.config.upper()
    phideg = phi_map[(pol_dir, args.phi)]
    base_args["phideg"] = phideg
    base_args["snapact"] = "snap_para" if pol_dir == "PARA" else "snap_perp"

    damage = DiamondDamageModel()
    dose = args.dose

    offsets = {
        "phi0": np.deg2rad(np.random.uniform(-2, 2)),
        "Bv":   np.random.normal(0, 5e-3),
        "Bh":   np.random.normal(0, 5e-3),
        "Theta": np.random.normal(30e-3, 10e-3),
        "Phi":  np.deg2rad(np.random.uniform(0, 360))
    }
    print("\nRandom goniometer–diamond offsets (radians):")
    for k, v in offsets.items():
        print(f"  {k:>5} = {v:.5f}")

    # Fixed wobble parameters from Jefferson Lab study (mrad)
    yaw_wobble_amp = 0.0004
    yaw_wobble_period = 0.05
    pitch_wobble_amp = 0.0006
    pitch_wobble_period = 0.01
    yaw_phase = np.random.uniform(0, 2*np.pi)
    pitch_phase = np.random.uniform(0, 2*np.pi)

    make_beamtilt(base_args)
    status, output = snap_crystal_orientation(base_args)
    thetah_0, thetav_0 = map(lambda s: float(s.strip()), output[0].decode().split(","))

    yaw_state = AxisBacklashState(pos_true=thetah_0, pos_readback=thetah_0)
    pitch_state = AxisBacklashState(pos_true=thetav_0, pos_readback=thetav_0)

    nsteps = args.nsteps
    print(f"\nSimulating {nsteps} stochastic nudges (goniometer + beam) using {args.nproc} cores ...")

    params_list = []
    for i in range(nsteps):
        # Random step direction for goniometer
        yaw_target = yaw_state.pos_true + np.random.choice([-1, 1]) * args.step
        pitch_target = pitch_state.pos_true + np.random.choice([-1, 1]) * args.step

        # Random beam tilts (simulate stochastic beam steering)
        beam_delh = np.random.normal(0.0, 0.002)  # horizontal beam tilt [mrad]
        beam_delv = np.random.normal(0.0, 0.001)  # vertical beam tilt [mrad]

        yaw_true, yaw_readback = step_with_backlash(yaw_target, args.step, args.backlash_n, yaw_state)
        pitch_true, pitch_readback = step_with_backlash(pitch_target, args.step, args.backlash_n, pitch_state)

        # Apply wobble to true angles
        yaw_true_wobble = apply_wobble(yaw_true, yaw_wobble_amp, yaw_wobble_period, yaw_phase)
        pitch_true_wobble = apply_wobble(pitch_true, pitch_wobble_amp, pitch_wobble_period, pitch_phase)

        params_list.append((i, nsteps,
                            yaw_true_wobble, pitch_true_wobble,
                            yaw_readback, pitch_readback,
                            beam_delh, beam_delv,
                            base_args, dose, damage, offsets))

    # Parallel execution
    t0 = time.time()
    results = []
    with mp.Pool(processes=min(args.nproc, os.cpu_count())) as pool:
        for res in tqdm(pool.imap_unordered(compute_peak_energy, params_list),
                        total=len(params_list), desc="Simulating", ncols=90):
            results.append(res)
    print(f"\nCompleted {len(results)} stochastic steps in {time.time() - t0:.2f} s")

    outfilename = f"stochastic_goniometer_beam_{pol_dir}_{args.phi.replace('/', '-')}_backlash_wobble.csv"
    with open(outfilename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "yaw_readback", "pitch_readback",
            "yaw_true", "pitch_true",
            "beam_delh_eff", "beam_delv_eff",
            "thetah_eff", "thetav_eff",
            "beamx", "beamy", "peak_energy",
            "Ga_deg", "Gv_deg", "Gh_deg"
        ])
        writer.writerows(results)
    print(f"Saved results to {outfilename}\n")


if __name__ == "__main__":
    main()
