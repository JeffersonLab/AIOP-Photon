#!/usr/bin/env python3
import os
import csv
import numpy as np
import ROOT
import time
import argparse
from spotfinder import make_beamtilt, fill_cobrems_polarintensity, snap_crystal_orientation

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
args_cli = parser.parse_args()

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
    args["thetah"] = thetah
    args["thetav"] = thetav

    # Apply beam jitter
    beamx = np.random.normal(args["xoffset"], args["beamx_noise"])
    beamy = np.random.normal(args["yoffset"], args["beamy_noise"])
    args["xoffset"] = beamx
    args["yoffset"] = beamy

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
    y_total = y_coh + y_inc

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
    writer.writerow(["thetah", "thetav", "beamx", "beamy", "phideg", "peak_energy"])

    phideg_rad = np.rad2deg(phideg)
    
    for c in c_vals:
        if pol_dir=="PERP":
            thetah = thetah_0 - c*np.cos(phideg_rad)
            thetav = thetav_0 + c*np.sin(phideg_rad)
        else:
            thetah = thetah_0 + c*np.sin(phideg_rad)
            thetav = thetav_0 + c*np.cos(phideg_rad)            
        peak_energy, beamx, beamy = compute_peak_energy(thetah, thetav, base_args)
        writer.writerow([thetah, thetav, beamx, beamy, base_args['phideg'], peak_energy])
        print(f"ThetaH={thetah:.5f}, ThetaV={thetav:.5f}, Phi={base_args['phideg']:3}° → Peak = {peak_energy:.5f} GeV")
