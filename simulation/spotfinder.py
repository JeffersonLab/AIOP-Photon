#!/usr/bin/env python3
#
# spotfinder.py - wsgi script that provides a visual interface
#                 to the UConn database of X-ray rocking curves
#                 of GlueX diamond radiators, together with means
#                 to rotate and shift the radiator across the beam
#                 in search of an optimum location in terms of the
#                 coherent enhancement and linear polarization of
#                 the collimated bremsstrahlung photon beam.
#
# author: richard.t.jones at uconn.edu
# version: june 3, 2022
#
# usage: direct a browser to the following URL,
#    https://gryphn.phys.uconn.edu/halld/diamonds/spotfinder

import os
import sys
import shutil
import subprocess
from urllib import parse as urlparse
import numpy as np
import pickle
import base64
import random
import time

os.environ["CPATH"] = "/home/patrick/miniconda3/envs/spotfinder-new/include"
os.environ["LD_LIBRARY_PATH"] = "/home/patrick/miniconda3/envs/spotfinder-new/lib"

# The following variables MUST be customized for the local site
docroot = "/home/patrick/AIOP-Photon"
topdir = "simulation"
tmpdir = "simulation/tmp"
self_script = "http://localhost/simulation"

sys.path.append(f"{docroot}{topdir}")
import cobrems_worker
ROOT = cobrems_worker.ROOT

radiator_names = ["JD70-103", "JD70-106", "JD70-107", "JD70-109"]
radiator_views = {"front": "_front_view.png", "back": "_back_view.png"}

mElectron = 0.51099895e-3 # GeV/c^2
qDiamond = 9.8e-6 # GeV/c

def draw_beamspot(args, size_px=600):
   """
   Creates an image of the electron beam spot at the Hall D radiator,
   saves it in a graphics file, and returns a relative path to the file.
   The native size of the image is size_px x size_px, which covers an
   area 20mm x 20mm centered on the beam. Conversion of background
   in the image from white to transparent is done using imagemagick.
   """
   img1 = f"beamspotw_{args['xsigma']},{args['ysigma']},{args['xycorr']},{size_px},{args['xyresol']}.png"
   img2 = f"beamspott_{args['xsigma']},{args['ysigma']},{args['xycorr']},{size_px},{args['xyresol']}.png"
   if not os.path.exists(img2):
      hlist = ROOT.beamspot(args['xsigma'], args['ysigma'], args['xycorr'], args['xyresol'])
      g = ROOT.draw_beam_spot(hlist[0], size_px, f"{docroot}{tmpdir}/{img1}")
      cmd = ["convert", f"{docroot}{tmpdir}/{img1}", 
             "-transparent", "white", f"{docroot}{tmpdir}/{img2}"]
      rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   return f"{tmpdir}/{img2}"

def unique_radiator_string(args):
   """
   Generates a unique string out of the contents of args
   related to the electron beam at the entrance to the radiator.
   """
   return (f"{args['radname']},{args['iradview']}" +
           f",{args['ebeam']},{args['penergy0']},{args['penergy1']}" +
           f",{args['peresol']},{args['ibeam']},{args['xyresol']}" +
           f",{args['xsigma']},{args['ysigma']},{args['xycorr']}" +
           f",{args['xoffset']},{args['yoffset']},{args['phideg']}" +
           f",{args['tiltrange']},{args['tiltresol']}")

def unique_collimator_string(args):
   """
   Generates a unique string out of the contents of args
   related to the photon beam at the entrance to the collimator.
   """
   return (f"{args['thetah']},{args['thetav']}" +
           f",{args['vspotrms']},{args['emittance']},{args['ebeamrms']}" +
           f",{args['radthick']},{args['coldiam']},{args['coldist']}")

def make_beamtilt(args):
   """
   Creates a 2D histogram of the horizontal and vertical tilt angles of
   the crystal planes of the radiator identificed by radname, radview 
   weighted by the electron beam spot centered at xoffset,yoffset from
   the center of the crystal, with rms sizes xsigma,ysigma and correlation
   coefficient xycorr.
   """
   hfile = "tiltspot_" + unique_radiator_string(args) + ".root"
   hpath = f"{docroot}{tmpdir}/{hfile}"
   #try:
   #   hlist = ROOT.TObjArray()
   #   f = ROOT.TFile(hpath)
   #   for h in ("tiltspot", "beamspot4topo", "topo4beamspot", "topo4beamspot2"):
   #      h2 = f.Get(h)
   #      h2.SetDirectory(0)
   #      hlist.Add(h2)
   #except:
   #   hlist = ROOT.tiltspot(args['radname'], args['iradview'],
   #                         args['xoffset'], args['yoffset'], args['phideg'],
   #                         args['xsigma'], args['ysigma'], args['xycorr'],
   #                         args['tiltresol'], args['tiltrange'], args['tiltrange'])
   #   f = ROOT.TFile(hpath, "create")
   #   hlist.Write()
   #   f.Close()
   hlist = ROOT.TObjArray()
   if os.path.exists(hpath):
      f = ROOT.TFile(hpath)
      valid = True
      #for h in ("cobrems_intensity", "amorph_intensity"):
      for h in ("tiltspot", "beamspot4topo", "topo4beamspot", "topo4beamspot2"):
         h2 = f.Get(h)
         if h2:
            h2.SetDirectory(0)
            hlist.Add(h2)
         else:
            valid = False
            break  # fall back to recompute if something's wrong
      if valid:
         return hlist  # everything loaded successfully

   # fallback if file doesn't exist or data is missing
   #htilt = make_beamtilt(args)
   #hlist = fill_cobrems_intensity(args, nsamples, htilt[0])
   #hamor = ROOT.amorph_intensity(args['ebeam'], args['ibeam'],
   #                        args['peresol'], args['penergy0'], 
   #                        args['penergy1'])
   #hlist.Add(hamor[0])
   #hlist.Add(htilt[0])
   hlist = ROOT.tiltspot(args['radname'], args['iradview'],
                          args['xoffset'], args['yoffset'], args['phideg'],
                          args['xsigma'], args['ysigma'], args['xycorr'],
                          args['tiltresol'], args['tiltrange'], args['tiltrange'])
   f = ROOT.TFile(hpath, "create")
   hlist.Write()
   f.Close()

   return hlist


def safe_filename(s):
   """
   Replaces commas with underscores in filenames.
   """
   return s.replace(",", "_").replace(" ", "").replace(".", "p").replace("-", "m")

def make_cobrems_polarintensity(args, nsamples=100):
    """
    Compute the coherent bremsstrahlung spectra entirely in memory.
    No ROOT files are created or loaded from disk.
    Returns: TObjArray with [polar_intensity, cobrems_intensity, amorphous_intensity, tiltspot]
    """
    hlist = ROOT.TObjArray()

    # Compute beam tilt histogram for this position
    htilt = make_beamtilt(args)  # returns TObjArray of [tiltspot, ...]
    tilt_hist = htilt[0]

    # Fill coherent + polarized spectra by sampling the tiltspot
    hcore = fill_cobrems_polarintensity(args, nsamples, tilt_hist)

    # Fill amorphous background (unpolarized, no crystal structure)
    hamor = ROOT.amorph_intensity(args["ebeam"], args["ibeam"],
                                  args["peresol"], args["penergy0"],
                                  args["penergy1"])

    # Add histograms to list
    for h in hcore: hlist.Add(h)
    hlist.Add(hamor[0])     # index 2
    hlist.Add(tilt_hist)    # index 3 (optional: might be unused)

    return hlist

def make_cobrems_intensity(args, nsamples=100):
    """
    Compute unpolarized coherent bremsstrahlung intensity in memory.
    Returns: TObjArray with [cobrems_intensity, amorphous_intensity, tiltspot]
    """
    hlist = ROOT.TObjArray()

    # Compute 2D tiltspot histogram
    htilt = make_beamtilt(args)  # returns TObjArray
    tilt_hist = htilt[0]

    # Fill coherent intensity spectrum by sampling from tiltspot
    hcore = fill_cobrems_intensity(args, nsamples, tilt_hist)

    # Fill amorphous background spectrum (no crystal structure)
    hamor = ROOT.amorph_intensity(args["ebeam"], args["ibeam"],
                                  args["peresol"], args["penergy0"],
                                  args["penergy1"])

    # Add all histograms to output list
    for h in hcore: hlist.Add(h)
    hlist.Add(hamor[0])     # index 1
    hlist.Add(tilt_hist)    # index 2 (optional)

    return hlist

 

def make_projection(prof):
   """
   Makes a TH1D projection of a TProfile prof.
   """
   iprojection_index = random.randint(0, 1<<64)
   return prof.ProjectionX(f"px_{iprojection_index}")

# def fill_cobrems_intensity(args, nsamples, htilt, nsplit=1, batchsize=500):
#    """
#    Create and fill a histogram of the collimated coherent bremsstrahlung
#    beam intensity as a function of photon energy. This is implemented as
#    a sum over tilt angles of the crystal sampled randomly from the htilt
#    2D histogram. Computation of the intensity spectra for individual sets
#    of tilt angles is offloaded to a parallel processing engine powered by
#    Celery. It requires that separate worker processes be running somewhere
#    on the cluster, communicating with this script through the rabbitmq
#    message broker daemon running on localhost. The result is returned as
#    a 1D profile histogram of the collimated photon beam rate spectrum
#    under the conditions specified in args. Performance is limited by the
#    number of available nodes running the celery workers.
#    """
#    thetah_ref = float(args['thetah'])
#    thetav_ref = float(args['thetav'])
#    penergy0_ref = float(args['penergy0'])
#    penergy1_ref = float(args['penergy1'])
#    pesplit = (penergy1_ref - penergy0_ref) / nsplit
#    thetah_mr = np.array([0], dtype=float)
#    thetav_mr = np.array([0], dtype=float)
#    htot = 0
#    for b in range(0, nsamples, batchsize):
#       bmax = min(b + batchsize, nsamples)
#       procs = []
#       for i in range(b, bmax):
#          htilt.GetRandom2(thetah_mr, thetav_mr)
#          args['thetah'] = thetah_ref + thetah_mr[0]
#          args['thetav'] = thetav_ref + thetav_mr[0]
#          for j in range(nsplit):
#             args['penergy0'] = penergy0_ref + j * pesplit
#             args['penergy1'] = penergy0_ref + (j+1) * pesplit
#             procs.append(cobrems_worker.fill_intensity_hist.delay(args))
#       retries = 0
#       while len(procs) > 0:
#          pending = []
#          for proc in procs:
#             if proc.ready():
#                msg = proc.get()
#                h = pickle.loads(base64.b64decode(msg))
#                if htot:
#                   htot.Add(h)
#                else:
#                   htot = h
#             elif retries > 50:
#                proc.forget()
#             else:
#                pending.append(proc)
#          procs = pending
#          if len(procs) > 0:
#             time.sleep(1)
#             retries += 1
#    args['thetah'] = thetah_ref
#    args['thetav'] = thetav_ref
#    args['penergy0'] = penergy0_ref
#    args['penergy1'] = penergy1_ref
#    hlist = ROOT.TObjArray()
#    hlist.Add(htot)
#    return hlist

#def fill_cobrems_polarintensity(args, nsamples, htilt, nsplit=1, batchsize=500):
#   """
#   Create and fill a histogram of the collimated coherent bremsstrahlung
#   polarized intensity as a function of photon energy. This is implemented
#   as a sum over tilt angles of the crystal sampled randomly from the htilt
#   2D histogram. Computation of the intensity spectra for individual sets
#   of tilt angles is offloaded to a parallel processing engine powered by
#   Celery. It requires that separate worker processes be running somewhere
#   on the cluster, communicating with this script through the rabbitmq
#   message broker daemon running on localhost. The result is returned as
#   a 1D profile histogram of the collimated photon beam rate spectrum
#   under the conditions specified in args. Performance is limited by the
#   number of available nodes running the celery workers.
#   """
#   thetah_ref = float(args['thetah'])
#   thetav_ref = float(args['thetav'])
#   penergy0_ref = float(args['penergy0'])
#   penergy1_ref = float(args['penergy1'])
#   pesplit = (penergy1_ref - penergy0_ref) / nsplit
#   thetah_mr = np.array([0], dtype=float)
#   thetav_mr = np.array([0], dtype=float)
#   htot = [0,0]
#   for b in range(0, nsamples, batchsize):
#      bmax = min(b + batchsize, nsamples)
#      procs = []
#      for i in range(b, bmax):
#         htilt.GetRandom2(thetah_mr, thetav_mr)
#         args['thetah'] = thetah_ref + thetah_mr[0]
#         args['thetav'] = thetav_ref + thetav_mr[0]
#         for j in range(nsplit):
#            args['penergy0'] = penergy0_ref + j * pesplit
#            args['penergy1'] = penergy0_ref + (j+1) * pesplit
#            p0 = cobrems_worker.fill_intensity_hist.delay(args, polarized=0)
#            p1 = cobrems_worker.fill_intensity_hist.delay(args, polarized=1)
#            setattr(p0, 'unpolarized_or_polarized', 0)
#            setattr(p1, 'unpolarized_or_polarized', 1)
#            procs.append(p0)
#            procs.append(p1)
#      retries = 0
#      while len(procs) > 0:
#         pending = []
#         for proc in procs:
#            if proc.ready():
#               msg = proc.get()
#               h = pickle.loads(base64.b64decode(msg))
   #             p = proc.unpolarized_or_polarized
   #             if htot[p]:
   #                htot[p].Add(h)
   #             else:
   #                htot[p] = h
   #          #elif retries > 50:
   #          #   proc.forget()
   #          elif retries > 50:
   #             try:
   #                proc.revoke(terminate=True)
   #             except Exception:
   #                pass
   #          else:
   #             pending.append(proc)
   #       procs = pending
   #       if len(procs) > 0:
   #          time.sleep(1)
   #          retries += 1
   # args['thetah'] = thetah_ref
   # args['thetav'] = thetav_ref
   # args['penergy0'] = penergy0_ref
   # args['penergy1'] = penergy1_ref
   # hlist = ROOT.TObjArray()
   # hlist.Add(htot[1])
   # hlist.Add(htot[0])
   # return hlist

def fill_cobrems_intensity(args, nsamples, htilt, nsplit=1, batchsize=500):
    """
    Local version of fill_cobrems_intensity — runs fully in-process using ROOT,
    no Celery or RabbitMQ required.

    Computes the collimated coherent bremsstrahlung beam intensity as a function
    of photon energy. Returns: ROOT.TObjArray([htot]) where htot is a TH1D.

    Parameters
    ----------
    args : dict
        Beam and radiator parameters.
    nsamples : int
        Number of Monte Carlo samples drawn from htilt.
    htilt : ROOT.TH2
        2D histogram of horizontal/vertical tilt angles.
    nsplit : int
        Optional energy segmentation count.
    batchsize : int
        Unused here, kept for API compatibility.
    """

    thetah_ref = float(args["thetah"])
    thetav_ref = float(args["thetav"])
    penergy0_ref = float(args["penergy0"])
    penergy1_ref = float(args["penergy1"])
    pesplit = (penergy1_ref - penergy0_ref) / nsplit

    # allocate temporary variables
    thetah_mr = np.array([0.0])
    thetav_mr = np.array([0.0])

    htot = None  # running total histogram

    # main sampling loop
    for i in range(nsamples):
        # sample random tilt from the tiltspot histogram
        htilt.GetRandom2(thetah_mr, thetav_mr)
        args["thetah"] = thetah_ref + thetah_mr[0]
        args["thetav"] = thetav_ref + thetav_mr[0]

        # optionally split energy range
        for j in range(nsplit):
            args["penergy0"] = penergy0_ref + j * pesplit
            args["penergy1"] = penergy0_ref + (j + 1) * pesplit

            # call the compiled C++ method directly
            hlist = ROOT.cobrems_intensity(
                args["radname"], args["iradview"],
                args["ebeam"], args["ibeam"],
                args["xyresol"],
                args["thetah"], args["thetav"],
                args["xoffset"], args["yoffset"],
                args["phideg"],
                args["xsigma"], args["ysigma"], args["xycorr"],
                args["peresol"], args["penergy0"], args["penergy1"], 0
            )

            h = hlist[0]
            h.SetDirectory(0)

            if htot:
                htot.Add(h)
            else:
                htot = h.Clone()
                htot.SetDirectory(0)

    # restore nominal values
    args["thetah"] = thetah_ref
    args["thetav"] = thetav_ref
    args["penergy0"] = penergy0_ref
    args["penergy1"] = penergy1_ref

    # package output to match the original API
    hlist_out = ROOT.TObjArray()
    hlist_out.Add(htot)
    return hlist_out

   
def fill_cobrems_polarintensity(args, nsamples, htilt, nsplit=1, batchsize=500):
    """
    Local version – compute spectra directly with ROOT, no Celery.
    Returns TObjArray([polarized_hist, unpolarized_hist]).
    """
    thetah_ref = float(args['thetah'])
    thetav_ref = float(args['thetav'])
    penergy0_ref = float(args['penergy0'])
    penergy1_ref = float(args['penergy1'])
    pesplit = (penergy1_ref - penergy0_ref) / nsplit

    thetah_mr = np.array([0.0])
    thetav_mr = np.array([0.0])
    h_pol = 0
    h_unpol = 0

    for i in range(nsamples):
        htilt.GetRandom2(thetah_mr, thetav_mr)
        args['thetah'] = thetah_ref + thetah_mr[0]
        args['thetav'] = thetav_ref + thetav_mr[0]
        for j in range(nsplit):
            args['penergy0'] = penergy0_ref + j * pesplit
            args['penergy1'] = penergy0_ref + (j + 1) * pesplit

            # direct C++ call instead of Celery task
            hlist_pol = ROOT.cobrems_intensity(
                args['radname'], args['iradview'],
                args['ebeam'], args['ibeam'],
                args['xyresol'], args['thetah'], args['thetav'],
                args['xoffset'], args['yoffset'], args['phideg'],
                args['xsigma'], args['ysigma'], args['xycorr'],
                args['peresol'], args['penergy0'], args['penergy1'], 1)
            hlist_unpol = ROOT.cobrems_intensity(
                args['radname'], args['iradview'],
                args['ebeam'], args['ibeam'],
                args['xyresol'], args['thetah'], args['thetav'],
                args['xoffset'], args['yoffset'], args['phideg'],
                args['xsigma'], args['ysigma'], args['xycorr'],
                args['peresol'], args['penergy0'], args['penergy1'], 0)

            h_p = hlist_pol[0]
            h_u = hlist_unpol[0]
            if h_pol:
                h_pol.Add(h_p)
                h_unpol.Add(h_u)
            else:
                h_pol = h_p
                h_unpol = h_u

    args['thetah'] = thetah_ref
    args['thetav'] = thetav_ref
    args['penergy0'] = penergy0_ref
    args['penergy1'] = penergy1_ref

    out = ROOT.TObjArray()
    out.Add(h_pol)
    out.Add(h_unpol)
    return out


def get_form_var(var, pars, dtype, default, unit="", err=0):
   try:
      val = dtype(pars[var][0])
   except:
      if err and var in pars:
         err.append(f"invalid form value for {var}")
      val = default
   return val

def snap_crystal_orientation(args):
   Eedge = args['snapedge']
   hlist = make_beamtilt(args)
   thetah_offset = hlist[0].ProjectionX('px').GetMean()
   thetav_offset = hlist[0].ProjectionY('py').GetMean()
   Ebeam = args['ebeam']
   gamma = Ebeam / mElectron
   one_minus_beta2 = 1 / gamma**2
   one_minus_beta = one_minus_beta2 / 2
   Qsintheta = ((Ebeam * Eedge * one_minus_beta - qDiamond**2 / 2)
                / (Ebeam - Eedge))
   theta_tilt = np.arcsin(Qsintheta / qDiamond) * 1000
   theta_tip = 50 # a good choice, not unique
   for i in range(5):
      one_minus_beta = one_minus_beta2 / (2 - one_minus_beta)
   if args['snapact'] == 'snap_perp':
      output = [f"{theta_tip}, {theta_tilt - thetav_offset}".encode()]
   elif args['snapact'] == 'snap_para':
      output = [f"{theta_tilt - thetah_offset}, {theta_tip}".encode()]
   else:
      output = [f"0, 0".encode()]
   status = "200 OK"
   return status,output

def plot_beamspot_on_topograph(args, size_px=600):
   imgf = "topospot_" + unique_radiator_string(args) + ".png"
   imgpath = f"{docroot}{tmpdir}/{imgf}"
   if not os.path.exists(imgpath):
      hlist = make_beamtilt(args)
      g = ROOT.draw_beamspot_on_topo(hlist[1], hlist[2], size_px, imgpath)
   hfile = imgf[:-4] + ".root"
   output = [f"""
    <img src="{tmpdir}/{imgf}">
    Iteration number {args['niter']} for 
    <a href="{tmpdir}/{hfile}">{hfile}</a>
   """.encode()]
   status = "200 OK"
   return status,output

def plot_tilt_intensity_map(args, size_px=600):
   imgf = "tiltspot_" + unique_radiator_string(args) + ".png"
   imgpath = f"{docroot}{tmpdir}/{imgf}"
   if not os.path.exists(imgpath):
      hlist = make_beamtilt(args)
      g = ROOT.draw_beam_tilt(hlist[0], size_px, imgpath)
   hfile = imgf[:-4] + ".root"
   output = [f"""
    <img src="{tmpdir}/{imgf}">
    Iteration number {args['niter']} for
    <a href="{tmpdir}/{hfile}">{hfile}</a>
   """.encode()]
   status = "200 OK"
   return status,output

def plot_cobrems_intensity_spectrum(args, size_px=600):
   imgf = ("intensity_" + unique_radiator_string(args) +
           "_" + unique_collimator_string(args) + ".png")
   imgpath = f"{docroot}{tmpdir}/{imgf}"
   if not os.path.exists(imgpath):
      hlist = make_cobrems_intensity(args)
      htot = make_projection(hlist[0])
      hinc = make_projection(hlist[1])
      htot.Add(hinc)
      hmax = htot.GetMaximum()
      i0 = htot.FindBin(args['penergy0'])
      i1 = htot.FindBin(args['penergy1'])
      hsum = htot.Integral(i0, i1)
      binwidth = htot.GetXaxis().GetBinWidth(1)
      htot.SetStats(0)
      htot.GetXaxis().SetRange(i0, i1)
      htot.SetTitle(f"peak = {hmax:.3e}/s/GeV, integral = {hsum*binwidth:.3e}/s")
      g = ROOT.draw_cobrems_spectrum(htot, size_px, imgpath)
   hfile = imgf[:-4] + ".root"
   output = [f"""
    <img src="{tmpdir}/{imgf}">
    Iteration number {args['niter']} for
    <a href="{tmpdir}/{hfile}">{hfile}</a>
   """.encode()]
   status = "200 OK"
   return status,output

def plot_cobrems_enhancement_spectrum(args, size_px=600):
   imgf = ("enhancement_" + unique_radiator_string(args) +
           "_" + unique_collimator_string(args) + ".png")
   imgpath = f"{docroot}{tmpdir}/{imgf}"
   if not os.path.exists(imgpath):
      hlist = make_cobrems_intensity(args)
      henh = make_projection(hlist[0])
      hinc = make_projection(hlist[1])
      henh.Add(hinc)
      henh.Divide(hinc)
      hmax = henh.GetMaximum()
      i0 = henh.FindBin(args['penergy0'])
      i1 = henh.FindBin(args['penergy1'])
      hsum = henh.Integral(i0, i1)
      henh.SetStats(0)
      henh.GetXaxis().SetRange(i0, i1)
      henh.GetYaxis().SetTitle("collimated beam coherent enhancement")
      henh.SetTitle(f"peak = {hmax:.3f}, average = {hsum/(i1-i0+1):.3e}")
      g = ROOT.draw_cobrems_spectrum(henh, size_px, imgpath)
   hfile = imgf[:-4] + ".root"
   output = [f"""
    <img src="{tmpdir}/{imgf}">
    Iteration number {args['niter']} for
    <a href="{tmpdir}/{hfile}">{hfile}</a>
   """.encode()]
   status = "200 OK"
   return status,output

def plot_cobrems_polarization_spectrum(args, size_px=600):
   imgf = ("polarintensity_" + unique_radiator_string(args) +
           "_" + unique_collimator_string(args) + ".png")
   imgpath = f"{docroot}{tmpdir}/{imgf}"
   if not os.path.exists(imgpath):
      hlist = make_cobrems_polarintensity(args)
      hpol = make_projection(hlist[0])
      htot = make_projection(hlist[1])
      hinc = make_projection(hlist[2])
      htot.Add(hinc)
      hpol.Divide(htot)
      hmax = hpol.GetMaximum()
      i0 = hpol.FindBin(args['penergy0'])
      i1 = hpol.FindBin(args['penergy1'])
      hsum = hpol.Integral(i0, i1)
      hpol.SetStats(0)
      hpol.GetXaxis().SetRange(i0, i1)
      hpol.GetYaxis().SetTitle("linear polarization")
      hpol.SetTitle(f"peak = {hmax:.3e}, average = {hsum/(i1-i0+1):.3e}")
      g = ROOT.draw_cobrems_spectrum(hpol, size_px, imgpath)
   hfile = imgf[5:-4] + ".root"
   output = [f"""
    <img src="{tmpdir}/{imgf}">
    Iteration number {args['niter']} for
    <a href="{tmpdir}/{hfile}">{hfile}</a>
   """.encode()]
   status = "200 OK"
   return status,output

def process_request(env, pars):
   logmsg = []
   args = {}
   args["radname"] = get_form_var("radiator_name", pars, dtype=str, default=radiator_names[0], err=logmsg)
   args["radview"] = get_form_var("radiator_view", pars, dtype=str, default="front", err=logmsg)
   args['iradview'] = 1 if args['radview'] == "back" else 0
   args["ebeam"] = get_form_var("ebeam_energy", pars, dtype=float, default=11.6, unit="GeV", err=logmsg)
   args["penergy0"] = get_form_var("pbeam_energy_min", pars, dtype=float, default=4.5, unit="GeV", err=logmsg)
   args["penergy1"] = get_form_var("pbeam_energy_max", pars, dtype=float, default=7.0, unit="GeV", err=logmsg)
   args["peresol"] = get_form_var("pbeam_energy_resol", pars, dtype=float, default=0.02, unit="GeV", err=logmsg)
   args["ibeam"] = get_form_var("ebeam_current", pars, dtype=float, default=2.2, unit="uA", err=logmsg)
   args["xsigma"] = get_form_var("ebeam_xsigma", pars, dtype=float, default=1.0, unit="mm", err=logmsg)
   args["ysigma"] = get_form_var("ebeam_ysigma", pars, dtype=float, default=0.5, unit="mm", err=logmsg)
   args["xycorr"] = get_form_var("ebeam_xycorr", pars, dtype=float, default=-0.6, unit="", err=logmsg)
   args["xyresol"] = get_form_var("ebeam_xyresol", pars, dtype=float, default=0.02, unit="mm", err=logmsg)
   args["xoffset"] = get_form_var("rad_xoffset", pars, dtype=float, default=0, unit="mm", err=logmsg)
   args["yoffset"] = get_form_var("rad_yoffset", pars, dtype=float, default=0, unit="mm", err=logmsg)
   args["phideg"] = get_form_var("rad_phideg", pars, dtype=float, default=0, unit="deg", err=logmsg)
   args["thetah"] = get_form_var("rad_thetah", pars, dtype=float, default=0, unit="mr", err=logmsg)
   args["thetav"] = get_form_var("rad_thetav", pars, dtype=float, default=0, unit="mr", err=logmsg)
   args["snapact"] = get_form_var("snap_action", pars, dtype=str, default="off", err=logmsg)
   args["snapedge"] = get_form_var("snap_edge", pars, dtype=float, default=6.0, unit="GeV", err=logmsg)
   args["vspotrms"] = get_form_var("vspot_rms", pars, dtype=float, default=0.5, unit="mm", err=logmsg)
   args["radthick"] = get_form_var("rad_thickness", pars, dtype=float, default=50, unit="microns", err=logmsg)
   args["emittance"] = get_form_var("ebeam_emittance", pars, dtype=float, default=4.2e-9, unit="m.radians", err=logmsg)
   args["ebeamrms"] = get_form_var("ebeam_rms", pars, dtype=float, default=0.001, unit="GeV", err=logmsg)
   args["coldiam"] = get_form_var("collimator_diameter", pars, dtype=float, default=3.4, unit="mm", err=logmsg)
   args["coldist"] = get_form_var("collimator_distance", pars, dtype=float, default=76, unit="m", err=logmsg)
   args["tiltrange"] = get_form_var("rad_tilt_range", pars, dtype=float, default=1, unit="mr", err=logmsg)
   args["tiltresol"] = get_form_var("rad_tilt_resol", pars, dtype=float, default=0.01, unit="mr", err=logmsg)
   args["toplot"] = get_form_var("plot_type", pars, dtype=str, default="tilt", err=logmsg)
   spot_plot_options = 'checked="checked"' if args["toplot"] == "spot" else ""
   tilt_plot_options = 'checked="checked"' if args["toplot"] == "tilt" else ""
   intense_plot_options = 'checked="checked"' if args["toplot"] == "intense" else ""
   enhance_plot_options = 'checked="checked"' if args["toplot"] == "enhance" else ""
   polar_plot_options = 'checked="checked"' if args["toplot"] == "polar" else ""
   ROOT.cobrems.setBeamErms(args["ebeamrms"])
   ROOT.cobrems.setBeamEmittance(args["emittance"])
   ROOT.cobrems.setCollimatorSpotrms(args["vspotrms"] * 1e-3)
   ROOT.cobrems.setCollimatorDistance(args["coldist"])
   ROOT.cobrems.setCollimatorDiameter(args["coldiam"] * 1e-3)
   ROOT.cobrems.setTargetThickness(args["radthick"] * 1e-6)
   ROOT.cobrems.setTargetCrystal("diamond")
   args["niter"] = get_form_var("iteration", pars, int, 0, logmsg)
   if args["snapact"] != "off":
      return snap_crystal_orientation(args)
   elif args["niter"] == 0:
      pass
   elif args["toplot"] == "spot":
      return plot_beamspot_on_topograph(args)
   elif args["toplot"] == "tilt":
      return plot_tilt_intensity_map(args)
   elif args["toplot"] == "intense":
      return plot_cobrems_intensity_spectrum(args)
   elif args["toplot"] == "enhance":
      return plot_cobrems_enhancement_spectrum(args)
   elif args["toplot"] == "polar":
      return plot_cobrems_polarization_spectrum(args)
   img = draw_beamspot(args, size_px=600)
   cosp = np.cos(args["phideg"] * np.pi/180)
   sinp = np.sin(args["phideg"] * np.pi/180)
   output = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>diamond radiator spot finder tool</title>
<style>
.canvas-container {{
  width: 520px;
  height: 520px;
  position: relative;
  top: -50px;
  left: -30px;
  margin: 20px;
  z-index: -1;
}}
.canvas-frame {{
  width: 600px;
  height: 600px;
  position: absolute;
  top: 0;
  left: 0;
  z-index: -2;
}}
.canvas-overlay {{
  width: 280px;
  height: 280px;
  position: absolute;
  top: 0px;
  left: 0px;
  z-index: -3;
}}
.slider {{
  width: 100%;
}}
.collimation-form-modal {{
  display: none;
  background-color: white;
  width: 500px;
  height:350px;
  top: 20%;
  left: 15%;
  position: absolute;
  z-index: 3;
}}
.collimation-form-overlay {{
  position: fixed;
  display: none;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0,0,0,0.5);
  z-index: 2;
  cursor: pointer;
}}
.collimation-form-close {{
  position: relative;
  float: right;
  margin: 10px;
  font-size: 40px;
}}
.collimation-form-close:hover {{
  cursor: pointer;
}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/9.3.2/math.js"></script>
</head>
<body onload="animateRadiatorMotion()">
<h1>diamond radiator spot finder tool</h1>
<div style="position:relative; top: -15px;">
Richard Jones, University of Connecticut, June 2022
</div>
<p>
<form id="cobrems-controls">
<table>
<tr>
 <td>
  <label for="radiator_name">Choose a radiator:</label>
  <select name="radiator_name" id="radiator_name" form="cobrems-controls">
   """.encode()]
   for name in radiator_names:
      if name == args["radname"]:
         sel = 'selected="selected"'
      else:
         sel = ''
      output += [f"""
    <option {sel} value="{name}">{name}</option>""".encode()]
   output += [f"""
  </select>
  <label for="radiator_view">side facing upstream:</label>
  <select name="radiator_view" id="radiator_view" form="cobrems-controls">
   """.encode()]
   for view in radiator_views:
      if view == args["radview"]:
         sel = 'selected="selected"'
      else:
         sel = ''
      output += [f"""
    <option {sel} value="{view}">{view}</option>""".encode()]
   output += [f"""
  </select>
  <input type="submit" value="Update" form="cobrems-controls">
  <div class="canvas-container">
   <img class="canvas-frame" src="{img}" alt="beam spot image at radiator"/>
   <img class="canvas-overlay" src="{tmpdir}/{args['radname'] + radiator_views[args['radview']]}"
        id="radiator_image" style="transform: translate(120px,180px);" alt="radiator topographic image"/>
  </div>
  <div style="text-align: center; vertical-align: bottom;">
   <a href="#" class="collimation-form-open">Check / modify parameters related to photon beam collimation</a>
   <div class="collimation-form-overlay"></div>
   <div class="collimation-form-modal">
    <span class="collimation-form-close">&times;</span>
    <h2>Check / modify parameters related to photon beam collimation</h2>
    <table align="center"><tr>
      <td style="text-align: left;">virtual spot rms size at collimator:</td>
      <td><input style="float: right;" type="text" id="vspot_rms" name="vspot_rms" value="{args['vspotrms']}" size="3"></td>
      <td style="text-align: left;">(mm)</td>
    </tr></tr>
      <td style="text-align: left;">diamond radiator thickness:</td>
      <td><input style="float: right;" type="text" id="rad_thickness" name="rad_thickness" value="{args['radthick']}" size="3"></td>
      <td style="text-align: left;">(microns)</td>
    </tr></tr>
      <td style="text-align: left;">electron beam transverse emittance:</td>
      <td><input style="float: right;" type="text" id="ebeam_emittance" name="ebeam_emittance" value="{args['emittance']}" size="3"></td>
      <td style="text-align: left;">(m.radians)</td>
    </tr></tr>
      <td style="text-align: left;">electron beam energy width:</td>
      <td><input style="float: right;" type="text" id="ebeam_rms" name="ebeam_rms" value="{args['ebeamrms']}" size="3"></td>
      <td style="text-align: left;">(GeV)</td>
    </tr></tr>
      <td style="text-align: left;">collimator diameter:</td>
      <td><input style="float: right;" type="text" id="collimator_diameter" name="collimator_diameter" value="{args['coldiam']}" size="3"></td>
      <td style="text-align: left;">(mm)</td>
    </tr></tr>
      <td style="text-align: left;">radiator-collimator distance:</td>
      <td><input style="float: right;" type="text" id="collimator_distance" name="collimator_distance" value="{args['coldist']}" size="3"></td>
      <td style="text-align: left;">(m)</td>
    </tr></tr>
      <td></td><td><input style="margin: 20px;" type="submit" value="Update"></td>
    </tr></table>
   </div>
  </div>
 </td>
 <td style="vertical-align: top;">
  <table>
   <tr>
    <td colspan="2">
     <h3>electron beam properties</h3>
   </td><td>
    <input style="float: right;" type="submit" value="Update">
   </td>
   </tr><tr>
    <td style="text-align: left;"><label for="ebeam_current"></label>
    <input type="hidden" id="ebeam_current" name="ebeam_current" value="{args['ibeam']}" size="3">
    <label for="ebeam_energy">electron beam energy:</label></td>
    <td><input type="text" id="ebeam_energy" name="ebeam_energy" value="{args['ebeam']}" size="3">
    </td><td>(GeV)</td>
   <tr>
    <td style="text-align: left;"><label for="ebeam_xsigma">beam rms size along x:</label></td>
    <td><input type="text" id="ebeam_xsigma" name="ebeam_xsigma" value="{args['xsigma']}" size="3">
    </td><td>(mm)</td>
   </tr><tr>
    <td style="text-align: left;"><label for="ebeam_ysigma">beam rms size along y:</label></td>
    <td><input type="text" id="ebeam_ysigma" name="ebeam_ysigma" value="{args['ysigma']}" size="3">
    </td><td>(mm)</td>
   </tr><tr>
    <td style="text-align: left;"><label for="ebeam_xycorr">beam xy correlation:</label></td>
    <td><input type="text" id="ebeam_xycorr" name="ebeam_xycorr" value="{args['xycorr']}" size="3">
    </td><td>[-1,1]</td>
   </tr><tr>
    <td style="text-align: left;"><label for="ebeam_xyresol">beam spot resolution:</label></td>
    <td><input type="text" id="ebeam_xyresol" name="ebeam_xyresol" value="{args['xyresol']}" size="3">
    </td><td>(mm)</td>
   </tr><tr>
    <td style="text-align: left;"><label for="rad_xoffset">radiator x offset:</label></td>
    <td><input type="text" id="rad_xoffset" name="rad_xoffset" value="{args['xoffset']}" size="3" oninput="sliderOutputHandler('rad_xoffset')">
    </td><td>(mm)</td>
   </tr><tr>
    <td colspan="3"><div class="slider-container">
     <input type="range" min="-10" max="10" step="0.01" name="rad_xoffsets" value="{args['xoffset']}" class="slider" id="rad_xoffset-slider" oninput="sliderInputHandler('rad_xoffset')">
    </div></td>
   </tr><tr>
    <td style="text-align: left;"><label for="rad_yoffset">radiator y offset:</label></td>
    <td><input type="text" id="rad_yoffset" name="rad_yoffset" value="{args['yoffset']}" size="3" oninput="sliderOutputHandler('rad_yoffset')">
    </td><td>(mm)</td>
   </tr><tr>
    <td colspan="3"><div class="slider-container">
     <input type="range" min="-10" max="10" step="0.01" name="rad_yoffsets" value="{args['yoffset']}" class="slider" id="rad_yoffset-slider" oninput="sliderInputHandler('rad_yoffset')">
    </div></td>
   </tr><tr>
    <td style="text-align: left;"><label for="rad_phideg">radiator phi rotation:</label></td>
    <td><input type="text" id="rad_phideg" name="rad_phideg" value="{args['phideg']}" size="3" oninput="sliderOutputHandler('rad_phideg')">
    </td><td>(deg)</td>
   </tr><tr>
    <td colspan="3"><div class="slider-container">
     <input type="range" min="-180" max="180" step="1" name="rad_phidegs" value="{args['phideg']}" class="slider" id="rad_phideg-slider" oninput="sliderInputHandler('rad_phideg')">
    </div></td>
   </tr><tr>
    <td style="text-align: left;"><label for="rad_thetah">radiator horizontal tilt:</label></td>
    <td><input type="text" id="rad_thetah" name="rad_thetah" value="{args['thetah']}" size="3" oninput="sliderOutputHandler('rad_thetah')">
    </td><td>(mr)</td>
   </tr><tr>
    <td colspan="3"><div class="slider-container">
     <input type="range" min="-10" max="10" step="0.01" name="rad_thetahs" value="{args['thetah']}" class="slider" id="rad_thetah-slider" oninput="sliderInputHandler('rad_thetah')">
    </div></td>
   </tr><tr>
    <td style="text-align: left;"><label for="rad_thetav">radiator vertical tilt:</label></td>
    <td><input type="text" id="rad_thetav" name="rad_thetav" value="{args['thetav']}" size="3" oninput="sliderOutputHandler('rad_thetav')">
    </td><td>(mr)</td>
   </tr><tr>
    <td colspan="3"><div class="slider-container">
     <input type="range" min="-10" max="10" step="0.01" name="rad_thetavs" value="{args['thetav']}" class="slider" id="rad_thetav-slider" oninput="sliderInputHandler('rad_thetav')">
    </div></td>
   </tr><tr>
    <td style="text-align: left;">
    <label for="rad_tilt_range">radiator tilt range</label>
    <input type="hidden" id="rad_tilt_range" name="rad_tilt_range" value="{args['tiltrange']}">
    <label for="rad_tilt_resol">radiator tilt resolution</label>
    <input type="hidden" id="rad_tilt_resol" name="rad_tilt_resol" value="{args['tiltresol']}">
    <label for="snap_edge">snap primary edge to</label>
    <input type="button" value="Para" onclick="snap_para()">
    <input type="button" value="Perp" onclick="snap_perp()"> at:
    <input type="hidden" id="snap_action" name="snap_action" value="off">
    <td><input type="text" id="snap_edge" name="snap_edge" value="{args['snapedge']}" size="3">
    </td><td>(GeV)</td>
   </tr><tr>
    <td style="text-align: left; vertical-align: bottom;" height="50"><b>What to plot</b></td>
   </tr><tr>
    <td style="text-align: left;">
     <input type="radio" id="spot-plot" name="plot_type" value="spot" {spot_plot_options}>
     <label for="spot-plot">beam spot on crystal</label><br/>
     <input type="radio" id="tilt-plot" name="plot_type" value="tilt" {tilt_plot_options}>
     <label for="tilt-plot">crystal tilt distribution</label><br/>
     <input type="radio" id="intense-plot" name="plot_type" value="intense" {intense_plot_options}>
     <label for="intense-plot">photon beam intensity spectrum</label><br/>
     <input type="radio" id="enhance-plot" name="plot_type" value="enhance" {enhance_plot_options}>
     <label for="enhance-plot">photon beam enhancement spectrum</label><br/>
     <input type="radio" id="polar-plot" name="plot_type" value="polar" {polar_plot_options}>
     <label for="polar-plot">photon beam polarization spectrum</label><br/>
   </td></tr>
  </table>
 </td>
 <td>
  <div style="margin-left: 50px;">
   <label for="pbeam_energy_min">photon energy window: from </label>
   <input type="text" id="pbeam_energy_min" name="pbeam_energy_min" value="{args['penergy0']}" size="3">
   <label for="pbeam_energy_max"> to </label>
   <input type="text" id="pbeam_energy_max" name="pbeam_energy_max" value="{args['penergy1']}" size="3">
   <label for="pbeam_energy_resol"> in steps of </label>
   <input type="text" id="pbeam_energy_resol" name="pbeam_energy_resol" value="{args['peresol']}" size="3"> GeV
  </div>
  <div id="cobrems-spectrum-plots" iteration="0" style="width: 600px; height: 570px; vertical-align: top; margin-left: 30px;">
  </div>
 </td>
</tr>
</table>
</form>
    """.encode()]
   for msg in logmsg:
      output.append(f"<p>{msg}</p>".encode())
   output.append(f"""
<script>
var radiator_px_per_mm = 24.2;
var radiator_xshift_px = 160;
var radiator_yshift_px = 160;
var radiator_xcenter_px = 0;
var radiator_ycenter_px = 15;
const max_pending_requests = 3;
var pending_requests = 0;
const queued_requests = [];
function sliderInputHandler(id) {{
  var tel = document.getElementById(id);
  var sel = document.getElementById(id + '-slider');
  tel.value = sel.value;
  animateRadiatorMotion();
}}
function sliderOutputHandler(id) {{
  var tel = document.getElementById(id);
  var sel = document.getElementById(id + '-slider');
  sel.value = tel.value;
  animateRadiatorMotion();
}}
function animateRadiatorMotion() {{
  var snapact = document.getElementById("snap_action");
  snapact.setAttribute("value", "off");
  var xel = document.getElementById("rad_xoffset");
  var xoffset_px = +xel.value * radiator_px_per_mm + radiator_xshift_px;
  var yel = document.getElementById("rad_yoffset");
  var yoffset_px = -yel.value * radiator_px_per_mm + radiator_yshift_px;
  var pel = document.getElementById("rad_phideg");
  var phideg = pel.value;
  var rel = document.getElementById("radiator_image");
  var xform = "translate(" + xoffset_px + "px," + yoffset_px + "px)";
  xform += " rotate(" + phideg + "deg)";
  xform += " translate(" + radiator_xcenter_px + "px," + radiator_ycenter_px + "px)";
  rel.setAttribute("style", "transform: " + xform + ";");
  var elmnt = document.getElementById("cobrems-spectrum-plots");
  refreshSpectrumPlots();
}}
function refreshSpectrumPlots() {{
  var rname = document.getElementById("radiator_name").value;
  var rview = document.getElementById("radiator_view").value;
  var benergy = document.getElementById("ebeam_energy").value;
  var bxsigma = document.getElementById("ebeam_xsigma").value;
  var bysigma = document.getElementById("ebeam_ysigma").value;
  var bxycorr = document.getElementById("ebeam_xycorr").value;
  var bxyresol = document.getElementById("ebeam_xyresol").value;
  var penergy0 = document.getElementById("pbeam_energy_min").value;
  var penergy1 = document.getElementById("pbeam_energy_max").value;
  var peresol = document.getElementById("pbeam_energy_resol").value;
  var xoffset = document.getElementById("rad_xoffset").value;
  var yoffset = document.getElementById("rad_yoffset").value;
  var phideg = document.getElementById("rad_phideg").value;
  var thetah = document.getElementById("rad_thetah").value;
  var thetav = document.getElementById("rad_thetav").value;
  var snapact = document.getElementById("snap_action").value;
  var snapedge = document.getElementById("snap_edge").value;
  var tiltresol = document.getElementById("rad_tilt_resol").value;
  var tiltrange = document.getElementById("rad_tilt_range").value;
  var vspotrms = document.getElementById("vspot_rms").value;
  var radthick = document.getElementById("rad_thickness").value;
  var emittance = document.getElementById("ebeam_emittance").value;
  var ebeamrms = document.getElementById("ebeam_rms").value;
  var coldiam = document.getElementById("collimator_diameter").value;
  var coldist = document.getElementById("collimator_distance").value;
  var toplot = "undefined";
  var elist = document.getElementsByTagName("input");
  for (i = 0; i < elist.length; i++) {{
    if (elist[i].type == "radio" && elist[i].checked) {{
      toplot = elist[i].value;
    }}
  }}
  var elmnt = document.getElementById("cobrems-spectrum-plots");
  elmnt.innerHTML = `
  <table><tr><th width="500" height="500" style="text-align: center; vertical-align: middle;">
  waiting for response from server...<br>
  (coarser energy steps are faster)
  </td></tr></table>
  `;
  var niter = parseInt(elmnt.getAttribute("iteration")) + 1;
  var httpRequest = "{self_script}";
  httpRequest += `?radiator_name=${{rname}}`;
  httpRequest += `&radiator_view=${{rview}}`;
  httpRequest += `&ebeam_energy=${{benergy}}`;
  httpRequest += `&ebeam_xsigma=${{bxsigma}}`;
  httpRequest += `&ebeam_ysigma=${{bysigma}}`;
  httpRequest += `&ebeam_xycorr=${{bxycorr}}`;
  httpRequest += `&ebeam_xyresol=${{bxyresol}}`;
  httpRequest += `&pbeam_energy_min=${{penergy0}}`;
  httpRequest += `&pbeam_energy_max=${{penergy1}}`;
  httpRequest += `&pbeam_energy_resol=${{peresol}}`;
  httpRequest += `&rad_xoffset=${{xoffset}}`;
  httpRequest += `&rad_yoffset=${{yoffset}}`;
  httpRequest += `&rad_phideg=${{phideg}}`;
  httpRequest += `&rad_thetah=${{thetah}}`;
  httpRequest += `&rad_thetav=${{thetav}}`;
  httpRequest += `&snap_action=${{snapact}}`;
  httpRequest += `&snap_edge=${{snapedge}}`;
  httpRequest += `&rad_tilt_range=${{tiltrange}}`;
  httpRequest += `&rad_tilt_resol=${{tiltresol}}`;
  httpRequest += `&vspot_rms=${{vspotrms}}`;
  httpRequest += `&rad_thickness=${{radthick}}`;
  httpRequest += `&ebeam_emittance=${{emittance}}`;
  httpRequest += `&ebeam_rms=${{ebeamrms}}`;
  httpRequest += `&collimator_diameter=${{coldiam}}`;
  httpRequest += `&collimator_distance=${{coldist}}`;
  httpRequest += `&plot_type=${{toplot}}`;
  httpRequest += `&iteration=${{niter}}`;
  elmnt.setAttribute("iteration", niter.toString());
  elmnt.setAttribute("w3-include-html", httpRequest);
  includeHTML();
}}
function includeHTML() {{
  var z, i, elmnt, file, xhttp, serialnumber;
  /* Loop through a collection of all HTML elements: */
  z = document.getElementsByTagName("*");
  for (i = 0; i < z.length; i++) {{
    elmnt = z[i];
    /*search for elements with a certain atrribute:*/
    file = elmnt.getAttribute("w3-include-html");
    if (file) {{
      /* Make an HTTP request using the attribute value as the file name: */
      xhttp = new XMLHttpRequest();
      xhttp.onreadystatechange = function() {{
        if (this.readyState == 4) {{
          if (this.status == 200) {{
            elmnt.innerHTML = this.responseText;
          }}
          else if (this.status == 404) {{
            elmnt.innerHTML = "Page not found.";
          }}
          else if (this.status == 400) {{
            elmnt.innerHTML = `Bad Request: ${{file}}`;
          }}
          pending_requests -= 1;
          /* Remove the attribute, and call this function once more: */
          let attr_value = elmnt.getAttribute("w3-include-html");
          if (attr_value !== null) {{
            elmnt.setAttribute("w3-included-html", attr_value);
            elmnt.removeAttribute("w3-include-html");
            includeHTML();
          }}
        }}
      }}
      if (pending_requests > max_pending_requests) {{
        xhttp["this_request_url"] = file;
        queued_requests.push(xhttp);
        console.log(`queued request count is now ${{queued_requests.length}}`);
      }}
      else {{
        xhttp.open("GET", file, true);
        xhttp.send();
        pending_requests += 1;
        console.log(`pending request count is now ${{pending_requests}}`);
      }}
      return;
    }}
  }}
  if (queued_requests.length > 0) {{
    /* only the latest request is worth sending */
    xhttp = queued_requests.pop();
    queued_requests.length = 0;
    xhttp.open("GET", xhttp["this_request_url"], true);
    xhttp.send();
  }}
}}
function snap_para() {{
  var snapact = document.getElementById("snap_action");
  snapact.setAttribute("value", "snap_para");
  execute_snap();
}}
function snap_perp() {{
  var snapact = document.getElementById("snap_action");
  snapact.setAttribute("value", "snap_perp");
  execute_snap();
}}
function execute_snap() {{
  var rname = document.getElementById("radiator_name").value;
  var rview = document.getElementById("radiator_view").value;
  var benergy = document.getElementById("ebeam_energy").value;
  var bcurrent = document.getElementById("ebeam_current").value;
  var bxsigma = document.getElementById("ebeam_xsigma").value;
  var bysigma = document.getElementById("ebeam_ysigma").value;
  var bxycorr = document.getElementById("ebeam_xycorr").value;
  var bxyresol = document.getElementById("ebeam_xyresol").value;
  var penergy0 = document.getElementById("pbeam_energy_min").value;
  var penergy1 = document.getElementById("pbeam_energy_max").value;
  var peresol = document.getElementById("pbeam_energy_resol").value;
  var xoffset = document.getElementById("rad_xoffset").value;
  var yoffset = document.getElementById("rad_yoffset").value;
  var phideg = document.getElementById("rad_phideg").value;
  var thetah = document.getElementById("rad_thetah").value;
  var thetav = document.getElementById("rad_thetav").value;
  var snapact = document.getElementById("snap_action").value;
  var snapedge = document.getElementById("snap_edge").value;
  var tiltrange = document.getElementById("rad_tilt_range").value;
  var tiltresol = document.getElementById("rad_tilt_resol").value;
  var vspotrms = document.getElementById("vspot_rms").value;
  var radthick = document.getElementById("rad_thickness").value;
  var emittance = document.getElementById("ebeam_emittance").value;
  var ebeamrms = document.getElementById("ebeam_rms").value;
  var coldiam = document.getElementById("collimator_diameter").value;
  var coldist = document.getElementById("collimator_distance").value;
  var toplot = "undefined";
  var elist = document.getElementsByTagName("input");
  for (i = 0; i < elist.length; i++) {{
    if (elist[i].type == "radio" && elist[i].checked) {{
      toplot = elist[i].value;
    }}
  }}
  var httpRequest = "{self_script}";
  httpRequest += `?radiator_name=${{rname}}`;
  httpRequest += `&radiator_view=${{rview}}`;
  httpRequest += `&ebeam_energy=${{benergy}}`;
  httpRequest += `&ebeam_current=${{bcurrent}}`;
  httpRequest += `&ebeam_xsigma=${{bxsigma}}`;
  httpRequest += `&ebeam_ysigma=${{bysigma}}`;
  httpRequest += `&ebeam_xycorr=${{bxycorr}}`;
  httpRequest += `&ebeam_xyresol=${{bxyresol}}`;
  httpRequest += `&pbeam_energy_min=${{penergy0}}`;
  httpRequest += `&pbeam_energy_max=${{penergy1}}`;
  httpRequest += `&pbeam_energy_resol=${{peresol}}`;
  httpRequest += `&rad_xoffset=${{xoffset}}`;
  httpRequest += `&rad_yoffset=${{yoffset}}`;
  httpRequest += `&rad_phideg=${{phideg}}`;
  httpRequest += `&rad_thetah=${{thetah}}`;
  httpRequest += `&rad_thetav=${{thetav}}`;
  httpRequest += `&snap_action=${{snapact}}`;
  httpRequest += `&snap_edge=${{snapedge}}`;
  httpRequest += `&rad_tilt_range=${{tiltrange}}`;
  httpRequest += `&rad_tilt_resol=${{tiltresol}}`;
  httpRequest += `&vspot_rms=${{vspotrms}}`;
  httpRequest += `&rad_thickness=${{radthick}}`;
  httpRequest += `&ebeam_emittance=${{emittance}}`;
  httpRequest += `&ebeam_rms=${{ebeamrms}}`;
  httpRequest += `&collimator_diameter=${{coldiam}}`;
  httpRequest += `&collimator_distance=${{coldist}}`;
  httpRequest += `&plot_type=${{toplot}}`;
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {{
    if (this.readyState == 4) {{
      if (this.status == 200) {{
        var tilts = this.responseText.split(/[, ]+/);
        var tilth = Math.round(parseFloat(tilts[0]) * 100) / 100;
        var tiltv = Math.round(parseFloat(tilts[1]) * 100) / 100;
        var thetahel = document.getElementById('rad_thetah');
        var thetahsel = document.getElementById('rad_thetah-slider');
        var thetavel = document.getElementById('rad_thetav');
        var thetavsel = document.getElementById('rad_thetav-slider');
        thetahel.value = tilth;
        thetahsel.value = tilth;
        thetavel.value = tiltv;
        thetavsel.value = tiltv;
        animateRadiatorMotion();
      }}
      else if (this.status == 404) {{
        console.log("Page not found.");
      }}
      else if (this.status == 400) {{
        console.log(`Bad Request: ${{httpRequest}}`);
      }}
    }}
  }}
  xhttp.open("GET", httpRequest, true);
  xhttp.send();
}}
var lightboxOpenButtons = document.getElementsByClassName("collimation-form-open");
var lightboxCloseButtons = document.getElementsByClassName("collimation-form-close");
var lightboxModals = document.getElementsByClassName("collimation-form-modal");
var lightboxOverlays = document.getElementsByClassName("collimation-form-overlay");
for (let i = 0; i < lightboxOpenButtons.length; i++) {{
  lightboxOpenButtons[i].addEventListener('click',
    function() {{
      lightboxModals[i].style.display = 'block';    
      lightboxOverlays[i].style.display = 'block';
    }}
  )
}}
for (let i = 0; i < lightboxCloseButtons.length; i++) {{
  lightboxCloseButtons[i].addEventListener("click",
    function() {{
      lightboxModals[i].style.display = 'none';
      lightboxOverlays[i].style.display = 'none';
    }}
  )
}}
for (let i = 0; i < lightboxOverlays.length; i++) {{
  lightboxOverlays[i].addEventListener("click",
    function() {{
      lightboxModals[i].style.display = 'none';
      lightboxOverlays[i].style.display = 'none';
    }}
  )
}}
</script>
</body>
</html>
   """.encode())
   status = "200 OK"
   return status,output

def application(environ, start_response):
   """
   Within the mod_wsgi, processing of the HTTP GET request 
   enters here with environ containing the key information
   from the request, and start_response to be loaded with 
   the header that leads the output to be sent back to the
   client.
   """
   env = environ
   pars = urlparse.parse_qs(environ["QUERY_STRING"], keep_blank_values=True)
   status,output = process_request(env, pars)
   output_len = sum([len(out) for out in output])
   response_headers = [("Content-type", "text/html"),
                       ("Content-Length", str(output_len))]
   start_response(status, response_headers)
   return output

def test():
   """
   Exercise various features, look for bugs."
   """
   args = {}
   args["radname"] = "JD70-103"
   args["radview"] = "front"
   args['iradview'] = 1 if args['radview'] == "back" else 0
   args["ebeam"] = 11.6
   args["xsigma"] = 1.0
   args["ysigma"] = 0.5
   args["xycorr"] = -0.6
   args["xyresol"] = 0.02
   args["xoffset"] = 0
   args["yoffset"] = 0
   args["phideg"] = 0
   args["thetah"] = 0
   args["thetav"] = 0
   args["toplot"] = "spot"
   args["niter"] = 0
   h = draw_beamspot(args)
   print("draw_beamspot(args) returns ", h)
   print("draw_beamspot(args) returns ", h)
   print("draw_beamspot(args) returns ", h)
   print("draw_beamspot(args) returns ", h)
   print("draw_beamspot(args) returns ", h)
   print("draw_beamspot(args) returns ", h)

if not os.path.isdir(f"{docroot}/{tmpdir}"):
   os.mkdir(f"{docroot}/{tmpdir}", 0o700)

   results_to_cache = [
     "JD70-103_010_results.root",
     "JD70-103_020_results.root",
     "JD70-103_030_results.root",
     "JD70-103_040_results.root",
     "JD70-103_couples.root",
     "JD70-106_010_results.root",
     "JD70-106_020_results.root",
     "JD70-106_030_results.root",
     "JD70-106_040_results.root",
     "JD70-106_couples.root",
     "JD70-107_010_results.root",
     "JD70-107_020_results.root",
     "JD70-107_030_results.root",
     "JD70-107_040_results.root",
     "JD70-107_couples.root",
     "JD70-109_011_results.root",
     "JD70-109_020_results.root",
     "JD70-109_030_results.root",
     "JD70-109_040_results.root",
     "JD70-109_couples.root"]
   for froot in results_to_cache:
      if not os.path.isfile(f"{docroot}{tmpdir}/{froot}"):
         shutil.copy(f"{docroot}{topdir}/results/{froot}", f"{docroot}{tmpdir}/{froot}")
   
   for radname in radiator_names:
      for radview in radiator_views:
         fimg = f"{radname}{radiator_views[radview]}"
         if not os.path.isfile(f"{docroot}{tmpdir}/{fimg}"):
            shutil.copy(f"{docroot}{topdir}/{fimg}", f"{docroot}{tmpdir}/{fimg}")
