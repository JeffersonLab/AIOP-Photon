#!/usr/bin/env python3
import os
import csv
import numpy as np
import ROOT
import time
from spotfinder import make_beamtilt, fill_cobrems_polarintensity

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

# ==== USER INPUT: Set polarization configuration ====
pol_dir = "PARA"          # options: "PARA", "PERP"
orientation = "0/90"      # options: "0/90", "45/135"
pitch_0, yaw_0 = 1.5, 1.5 # initial values of pitch and yaw to sweep around
# ====================================================

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
}

# Precompute the tiltspot histogram once 
print("Precomputing beam tilt histogram...")
htilt = make_beamtilt(base_args)[0]


# Function to extract x-position of peak (in GeV)
def compute_peak_energy(thetah, thetav, args, polarized=0):

    args = base_args.copy()
    args["thetah"] = thetah
    args["thetav"] = thetav

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
    
    return peak_energy

# Sweep through angles
c_vals = np.linspace(0, 0.5, 50)

# Output CSV
outfilename = f"coherent_peaks_{pol_dir}_{orientation.replace('/', '-')}.csv"
with open(outfilename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["thetah", "thetav", "beamx", "beamy", "phideg", "peak_energy"])

    for c in c_vals:
        if pol_dir=="PERP":
            thetah = pitch_0 - c*np.cos(base_args['phideg'])
            thetav = yaw_0 + c*np.sin(base_args['phideg'])
        else:
            thetah = pitch_0 + c*np.sin(base_args['phideg'])
            thetav = yaw_0 + c*np.cos(base_args['phideg'])            
        peak_energy = compute_peak_energy(thetah, thetav, base_args)
        writer.writerow([thetah, thetav, base_args['xoffset'], base_args['yoffset'], base_args['phideg'], peak_energy])
        print(f"ThetaH={thetah:.5f}, ThetaV={thetav:.5f}, Phi={base_args['phideg']:3}° → Peak = {peak_energy:.5f} GeV")
