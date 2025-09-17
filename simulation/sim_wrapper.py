#!/usr/bin/env python3
import os
import csv
import numpy as np
import ROOT
import time
import argparse
from spotfinder import make_beamtilt, fill_cobrems_polarintensity, snap_crystal_orientation
from damage_model import DiamondDamageModel

# Command line parsing
parser = argparse.ArgumentParser(description="Simulate coherent bremsstrahlung spectrum vs tilt")
parser.add_argument('--edge', type=float, required=True,
                    help='Nominal coherent edge energy in GeV')
parser.add_argument('--config', type=str, choices=['PARA', 'PERP'], required=True,
                    help='Polarization configuration: PARA or PERP')
parser.add_argument('--phi', type=str, choices=['0/90', '45/135'], required=True,
                    help='Orientation configuration: 0/90 or 45/135')
parser.add_argument('--beamx', type=float, required=False,
                    help='Beam x position in mm')
parser.add_argument('--beamy', type=float, required=False,
                    help='Beam y position in mm')
parser.add_argument('--dose', type=float, required=False, help='Integrated beam current')
parser.add_argument('--damage-alpha', type=float, default=0.0,
                    help='Broadening growth per dose [MeV/(e-/cm^2)]')
parser.add_argument('--damage-beta1', type=float, default=0.0,
                    help='Linear peak shift per dose [MeV/(e-/cm^2)]')
parser.add_argument('--damage-beta2', type=float, default=0.0,
                    help='Quadratic peak shift per dose^2 [MeV/(e-/cm^2)^2]')
parser.add_argument('--damage-D0', type=float, default=float('inf'),
                    help='Coherent amplitude e-fold dose [e-/cm^2]')
parser.add_argument('--damage-sigma0', type=float, default=0.0,
                    help='Baseline blur sigma at zero dose [MeV]')
parser.add_argument('--beam-delh', type=float, default=0.0,
                    help='Electron beam horizontal tilt (same units as thetah)')
parser.add_argument('--beam-delv', type=float, default=0.0,
                    help='Electron beam vertical tilt (same units as thetav)')
args_cli = parser.parse_args()


damage = DiamondDamageModel(alpha=args_cli.damage_alpha,
                            beta1=args_cli.damage_beta1,
                            beta2=args_cli.damage_beta2,
                            D0=args_cli.damage_D0,
                            sigma0=args_cli.damage_sigma0)
dose = args_cli.dose  # e-/cm^2


nominal_edge = args_cli.edge
pol_dir = args_cli.config.upper()
orientation = args_cli.phi

# Load ROOT macros and shared libraries
ROOT.gSystem.AddDynamicPath(os.environ.get('COBREMS_WORKER', '.'))
ROOT.gSystem.Load("libboost_python3.so")
ROOT.gSystem.Load("CobremsGeneration_cc.so")
ROOT.gSystem.Load("rootvisuals_C.so")

# Initialize CobremsGeneration
ebeam = 11.7  # GeV
ibeam = 2.2   # uA
ROOT.cobrems = ROOT.CobremsGeneration(ebeam, ibeam)
ROOT.cobrems.setBeamErms(0.001)
ROOT.cobrems.setBeamEmittance(4.2e-9)
ROOT.cobrems.setCollimatorSpotrms(0.5e-3)
ROOT.cobrems.setCollimatorDistance(76.0)
ROOT.cobrems.setCollimatorDiameter(3.4e-3)
ROOT.cobrems.setTargetCrystal("diamond")
ROOT.cobrems.setTargetThickness(50e-6)

# Determine phideg from configuration
phi_map = {
    ("PARA", "0/90"): 0,
    ("PERP", "0/90"): 90,
    ("PARA", "45/135"): 45,
    ("PERP", "45/135"): 135
}
phideg = phi_map[(pol_dir, orientation)]

# Define simulation parameters
base_args = {
    'radname': "JD70-103",
    'iradview': 0,
    'ebeam': ebeam,
    'ibeam': ibeam,
    'xyresol': 0.01,
    'xoffset': -0.38,
    'yoffset': 3.13,
    'phideg': phideg,
    'xsigma': 1.0,
    'ysigma': 0.5,
    'xycorr': 0.42,
    'peresol': 0.02,
    'penergy0': 4.5,
    'penergy1': 11.0,
    "tiltrange": 1.0,
    "tiltresol": 0.01,
    "ebeamrms": 0.001,
    "emittance": 4.2e-9,
    "vspotrms": 0.5,    
    "coldiam": 3.4,
    "coldist": 76,
    "radthick": 50,
    "snapedge": nominal_edge,
    'snapact': 'snap_para' if pol_dir == 'PARA' else 'snap_perp',
    "beamx_noise": 0.1,  # mm
    "beamy_noise": 0.1,  # mm
}

#Add beam angle tile to the base args
base_args["beam_th"] = float(args_cli.beam_th or 0.0)
base_args["beam_tv"] = float(args_cli.beam_tv or 0.0)

#snap the thetah and thetav into place
status, output = snap_crystal_orientation(base_args)
thetah_str, thetav_str = output[0].decode().split(",")
thetah_0 = float(thetah_str.strip())
thetav_0 = float(thetav_str.strip())

# Precompute the tiltspot histogram once 
print("Precomputing beam tilt histogram...")
htilt = make_beamtilt(base_args)[0]


# Function to extract x-position of peak (in GeV)
def compute_peak_energy(thetah, thetav, args, polarized=0):

    args = base_args.copy()

    # Convert diamond angles + beam tilt -> effective relative angles
    thetah_eff = thetah - args.get("beam_th", 0.0)
    thetav_eff = thetav - args.get("beam_tv", 0.0)
    args["thetah"] = thetah_eff
    args["thetav"] = thetav_eff
    
    # Apply beam jitter
    #beamx = np.random.normal(args["xoffset"], args["beamx_noise"])
    #beamy = np.random.normal(args["yoffset"], args["beamy_noise"])
    #args["xoffset"] = beamx
    #args["yoffset"] = beamy

    print(args["thetah"])
    print(args["thetav"])    
    
    hlist = fill_cobrems_polarintensity(args, nsamples=100, htilt=htilt)
    
    h_coh = hlist[1]

    # Add amorphous (incoherent) intensity manually                                                                                                 
    h_incoh = ROOT.amorph_intensity(args["ebeam"], args["ibeam"],
                    args["peresol"], args["penergy0"], args["penergy1"])[0]

    h_coh.SetDirectory(0)
    h_incoh.SetDirectory(0)
    
    n_bins = h_coh.GetNbinsX()

    energy = np.array([h_coh.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_coh  = np.array([h_coh.GetBinContent(i)  for i in range(1, n_bins + 1)])
    y_inc  = np.array([h_incoh.GetBinContent(i) for i in range(1, n_bins + 1)])
    #y_total = y_coh + y_inc

    # Energy units: your ROOT hist x-axis is in GeV; the model expects MeV for σ and shifts.
    # Convert to MeV for the convolution physics, then back.
    E_GeV = energy
    E_MeV = 1000.0 * E_GeV
    y_coh_damaged = damage.apply(E_MeV, y_coh, dose)
    y_total = y_coh_damaged + y_inc

    with np.errstate(divide='ignore', invalid='ignore'):
        enhancement = np.where(y_inc != 0, y_coh / y_inc, 0)

    # Find peak                                                                                                                                     
    if np.any(np.isfinite(enhancement)):
        max_idx = np.nanargmax(enhancement)
        peak_energy = energy[max_idx]
        peak_enhancement = enhancement[max_idx]
    else:
        peak_energy = np.nan
        peak_enhancement = np.nan
    
    return peak_energy, beamx, beamy

# Sweep through angles
c_vals = np.linspace(0, 50, 50)

# Output CSV
outfilename = f"coherent_peaks_{pol_dir}_{orientation.replace('/', '-')}.csv"
with open(outfilename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["thetah", "thetav", "beam_delh", "beam_delv", "beamx", "beamy", "phideg", "peak_energy"])

    phideg_rad = np.rad2deg(phideg)
    
    for c in c_vals:
        if pol_dir=="PERP":
            thetah = thetah_0 - c*np.cos(phideg_rad)
            thetav = thetav_0 + c*np.sin(phideg_rad)
        else:
            thetah = thetah_0 + c*np.sin(phideg_rad)
            thetav = thetav_0 + c*np.cos(phideg_rad)            
        peak_energy, beamx, beamy = compute_peak_energy(thetah, thetav, base_args)

        writer.writerow([thetah, thetav, base_args.get("beam_delh",0.0), base_args.get("beam_delv",0.0),
                         beamx, beamy, base_args['phideg'], peak_energy])
        print(f"Diamond θH={thetah:.5f}, θV={thetav:.5f}, Beam(θH,θV)=({base_args.get('beam_delh',0.0):.5f},{base_args.get('beam_delv',0.0):.5f}), "
              f"Phi={base_args['phideg']:3}° → Peak = {peak_energy:.5f} GeV")
