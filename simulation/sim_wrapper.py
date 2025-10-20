#!/usr/bin/env python3
import os
import csv
import time
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
from damage_model import DiamondDamageModel
from spotfinder import make_beamtilt, snap_crystal_orientation

#Livingston offset equations (Eqs. 5–7)
def compute_goniometer_angles(phi, c, i, offsets):
    phi0, Bv, Bh, Theta, Phi = (
        offsets["phi0"], offsets["Bv"], offsets["Bh"], offsets["Theta"], offsets["Phi"]
    )
    Ga = phi - phi0
    Gv = c*np.cos(phi) - i*np.sin(phi) - Theta*np.cos(Ga + Phi) + Bv
    Gh = c*np.sin(phi) + i*np.cos(phi) - Theta*np.sin(Ga + Phi) + Bh
    return Ga, Gv, Gh


#Initialize ROOT inside worker
def init_root(base_args):
    import ROOT

    libdir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gSystem.AddDynamicPath(libdir)
    ROOT.gSystem.Load(os.path.join(libdir, "CobremsGeneration_cc.so"))
    ROOT.gSystem.Load(os.path.join(libdir, "rootvisuals_C.so"))

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
    (i, total, thetah, thetav, beam_delh0, beam_delv0, base_args,
     dose, damage, offsets) = params

    ROOT = init_root(base_args)

    # Update beam tilt for this run
    base_args["beam_delh0"] = beam_delh0
    base_args["beam_delv0"] = beam_delv0

    # Compute amorphous intensity once per worker
    h_incoh = ROOT.amorph_intensity(
        base_args["ebeam"], base_args["ibeam"],
        base_args["peresol"], base_args["penergy0"], base_args["penergy1"]
    )[0]
    h_incoh.SetDirectory(0)

    # Effective diamond-beam tilts (include jitter)
    sigma_h = base_args.get("beam_delh_sigma", 0.0)
    sigma_v = base_args.get("beam_delv_sigma", 0.0)
    jitter_h = np.random.normal(0.0, sigma_h)
    jitter_v = np.random.normal(0.0, sigma_v)
    beam_delh_eff = beam_delh0 + jitter_h
    beam_delv_eff = beam_delv0 + jitter_v

    thetah_eff = thetah + beam_delh_eff
    thetav_eff = thetav + beam_delv_eff
    base_args["thetah"], base_args["thetav"] = thetah_eff, thetav_eff

    # Beam position jitter
    beamx = np.random.normal(base_args["xoffset"], base_args["beamx_noise"])
    beamy = np.random.normal(base_args["yoffset"], base_args["beamy_noise"])
    base_args["xoffset"], base_args["yoffset"] = beamx, beamy

    # Direct ROOT call
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

    # Goniometer angles
    phi = np.deg2rad(base_args["phideg"])
    c, i_angle = thetav * 1e-3, thetah * 1e-3  # mrad → rad
    Ga, Gv, Gh = compute_goniometer_angles(phi, c, i_angle, offsets)

    return (thetah, thetav, beam_delh0, beam_delv0,
            beam_delh_eff, beam_delv_eff,
            thetah_eff, thetav_eff,
            beamx, beamy, peak_energy,
            np.rad2deg(Ga), np.rad2deg(Gv), np.rad2deg(Gh))


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Local coherent bremsstrahlung simulator with beam sweep")
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--config", type=str, choices=["PARA", "PERP"], required=True)
    parser.add_argument("--phi", type=str, choices=["0/90", "45/135"], required=True)
    parser.add_argument("--dose", type=float, default=0.0)
    parser.add_argument("--beam-range", type=float, nargs=3, metavar=("MIN", "MAX", "N"),
                        default=[-0.3, 0.3, 3], help="beam pitch/yaw range [mrad]")
    parser.add_argument("--diamond-range", type=float, nargs=3, metavar=("MIN", "MAX", "N"),
                        default=[0, 50, 50], help="diamond tilt range [mrad]")
    parser.add_argument("--nproc", type=int, default=12)
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
        "beam_delh_sigma": 0.2, "beam_delv_sigma": 0.1,
    }

    phi_map = {("PARA", "0/90"): 0, ("PERP", "0/90"): 90,
               ("PARA", "45/135"): 135, ("PERP", "45/135"): 45}
    pol_dir = args.config.upper()
    phideg = phi_map[(pol_dir, args.phi)]
    base_args["phideg"] = phideg
    base_args["snapact"] = "snap_para" if pol_dir == "PARA" else "snap_perp"

    damage = DiamondDamageModel()
    dose = args.dose

    # Random Livingston offsets
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

    print("\nPrecomputing beam tilt ...")
    htilt = make_beamtilt(base_args)[0]
    status, output = snap_crystal_orientation(base_args)
    thetah_0, thetav_0 = map(lambda s: float(s.strip()), output[0].decode().split(","))

    # Build diamond and beam grids
    diamond_start, diamond_stop, diamond_num = args.diamond_range
    beam_start, beam_stop, beam_num = args.beam_range
    diamond_vals = np.linspace(diamond_start, diamond_stop, int(diamond_num))
    beam_vals = np.linspace(beam_start, beam_stop, int(beam_num))
    total = len(diamond_vals) * len(beam_vals)**2

    # Prepare tasks
    phideg_rad = np.deg2rad(phideg)
    params_list = []
    idx = 0
    for c in diamond_vals:
        if pol_dir == "PARA":
            thetah = thetah_0 - c * np.cos(phideg_rad)
            thetav = thetav_0 + c * np.sin(phideg_rad)
        else:
            thetah = thetah_0 + c * np.sin(phideg_rad)
            thetav = thetav_0 + c * np.cos(phideg_rad)
        for bh in beam_vals:
            for bv in beam_vals:
                idx += 1
                params_list.append((idx, total, thetah, thetav, bh, bv,
                                    base_args, dose, damage, offsets))

    print(f"\nRunning {total} total simulations using {args.nproc} cores ...")
    t0 = time.time()
    results = []
    with mp.Pool(processes=min(args.nproc, os.cpu_count())) as pool:
        for res in tqdm(pool.imap_unordered(compute_peak_energy, params_list),
                        total=len(params_list), desc="Simulating", ncols=90):
            results.append(res)
    print(f"\nCompleted {len(results)} points in {time.time() - t0:.2f} s")

    outfilename = f"coherent_peaks_{pol_dir}_{args.phi.replace('/', '-')}.csv"
    with open(outfilename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "thetah", "thetav",
            "beam_delh0", "beam_delv0",
            "beam_delh_eff", "beam_delv_eff",
            "thetah_eff", "thetav_eff",
            "beamx", "beamy", "peak_energy",
            "Ga_deg", "Gv_deg", "Gh_deg"
        ])
        writer.writerows(results)
    print(f"Saved results to {outfilename}\n")


if __name__ == "__main__":
    main()
