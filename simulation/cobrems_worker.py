#!/usr/bin/env python3
#
# cobrems_worker.py - Celery tasks for parallel computing of coherent
#                     bremsstrahlung spectra, used by the wsgi script
#                     spotfinder.py for real-time computing of coherent
#                     bremsstrahlung photon beam spectra.
#
# author: richard.t.jones at uconn.edu
# version: july 14, 2022

import os
import pickle
import base64
import random
import time
import numpy as np

import ROOT
ROOT.gSystem.AddDynamicPath(os.environ['COBREMS_WORKER'])
ROOT.gSystem.AddDynamicPath("/usr/lib64")
ROOT.gSystem.Load("libboost_python3.so")
ROOT.gSystem.Load("CobremsGeneration_cc.so")
ROOT.gSystem.Load("rootvisuals_C.so")
#to stop ROOT/Cling from using the conflicting Clang headers, tell ROOT to ignore its internal cling headers by unsetting the relevant include path.
ROOT.gSystem.AddIncludePath("-isystem /home/patrick/miniconda3/envs/spotfinder-new/include")
ROOT.gSystem.AddIncludePath("-isystem /home/patrick/miniconda3/envs/spotfinder-new/include/c++/13.3.0")
ROOT.gSystem.AddIncludePath("-isystem /home/patrick/miniconda3/envs/spotfinder-new/include/c++/13.3.0/x86_64-conda-linux-gnu")
ROOT.gSystem.AddIncludePath("-isystem /usr/include")  # Optional: For system headers


from celery import Celery

app = Celery("cobrems_worker", backend="rpc://",
        broker=f"amqp://admin:P3i@tet!@localhost//")
app.conf.update(result_expires=1200)

@app.task
def fill_random_hist(pid, nrand):
   """
   Example worker task for exercising ROOT object exchange.
   """
   h = ROOT.TH1D(f"rando{pid}", f"Rando Brando {pid}", 100, -1, 1)
   for r in range(nrand):
      h.Fill((pid + r) / nrand)
   return base64.b64encode(pickle.dumps(h)).decode()

@app.task
def fill_intensity_hist(args, polarized=0):
   """
   Compute the coherent bremsstrahlung intensity spectrum under the
   conditions specified in args, and return the result in the form
   of a 1D root histogram. This method runs in parallel within the
   celery task framework.
   """
   h = ROOT.gROOT.FindObject("intensity")
   if h:
      h.Delete()
   ROOT.cobrems.setBeamErms(args["ebeamrms"])
   ROOT.cobrems.setBeamEmittance(args["emittance"])
   ROOT.cobrems.setCollimatorSpotrms(args["vspotrms"] * 1e-3)
   ROOT.cobrems.setCollimatorDistance(args["coldist"])
   ROOT.cobrems.setCollimatorDiameter(args["coldiam"] * 1e-3)
   ROOT.cobrems.setTargetThickness(args["radthick"] * 1e-6)
   ROOT.cobrems.setTargetCrystal("diamond")
   hlist = ROOT.cobrems_intensity(args['radname'], args['iradview'],
                                  args['ebeam'], args['ibeam'], 
                                  args['xyresol'],
                                  args['thetah'], args['thetav'],
                                  args['xoffset'], args['yoffset'],
                                  args['phideg'],
                                  args['xsigma'], args['ysigma'],
                                  args['xycorr'], args['peresol'],
                                  args['penergy0'], args['penergy1'],
                                  polarized)
   return base64.b64encode(pickle.dumps(hlist[0])).decode()

def test_intensity_hist(nsamples=200, nsplit=1, batchsize=200, args={}):
   args['radname'] = "JD70-103"
   args['iradview'] = 0
   args['ebeam'] = 11.7
   args['ibeam'] = 2.2
   args['xyresol'] = 0.01
   args['thetah'] = -1.675
   args['thetav'] = 250
   args['xoffset'] = -0.38
   args['yoffset'] = 3.13
   args['phideg'] = 135
   args['xsigma'] = 1.0
   args['ysigma'] = 0.5
   args['xycorr'] = 0.42
   args['peresol'] = 0.02
   args['penergy0'] = 4.5
   args['penergy1'] = 7.0
   args["ebeamrms"] = 0.001
   args["emittance"] = 4.2e-9
   args["vspotrms"] = 0.5e-3
   args["coldist"] = 76
   args["coldiam"] = 3.4e-3
   args["radthick"] = 50e-6
   thetah_ref = float(args['thetah'])
   thetav_ref = float(args['thetav'])
   thetah_mr = np.array([0], dtype=float)
   thetav_mr = np.array([0], dtype=float)
   penergy0_ref = args['penergy0']
   penergy1_ref = args['penergy1']
   pesplit = (penergy1_ref - penergy0_ref) / nsplit
   htot = 0
   htilt = ROOT.gROOT.FindObject("tilttest")
   if not htilt:
      htilt = ROOT.TH2D("tilttest", "", 100, 0, 1, 100, 0, 1)
      for i in range(0,100):
         for j in range(0,100):
            rho2 = ((i - 50) / 11.2)**2 + ((j - 50) / 14.9)**2
            htilt.SetBinContent(i+1, j+1, np.exp(-0.5 * rho2))
   start = time.time()
   for b in range(0, nsamples, batchsize):
      bmax = min(b + batchsize, nsamples)
      procs = []
      for i in range(b, bmax):
         htilt.GetRandom2(thetah_mr, thetav_mr)
         args['thetah'] = thetah_ref + thetah_mr[0]
         args['thetav'] = thetav_ref + thetav_mr[0]
         for j in range(nsplit):
            args['penergy0'] = penergy0_ref + j * pesplit
            args['penergy1'] = penergy0_ref + (j+1) * pesplit
            procs.append(fill_intensity_hist.delay(args))
      print(f"collecting results from {len(procs)} tasks")
      while len(procs) > 0:
         pending = []
         for proc in procs:
            if proc.ready():
               msg = proc.get()
               h = pickle.loads(base64.b64decode(msg))
               if htot:
                  htot.Add(h)
               else:
                  htot = h
            else:
               pending.append(proc)
         print(f"batch {b} finished with {len(pending)} still pending")
         procs = pending
         if len(procs) > 0:
            time.sleep(1)
   args['thetah'] = thetah_ref
   args['thetav'] = thetav_ref
   args['penergy0'] = penergy0_ref
   args['penergy1'] = penergy1_ref
   hlist = ROOT.TObjArray()
   hlist.Add(htot)
   print("wall time used was", time.time() - start)
   return hlist

def test_random_hist(nprocs=100000, nrand=100000, batchsize=900):
   for b in range(0, nprocs, batchsize):
      bmax = min(b + batchsize, nprocs)
      procs = [fill_random_hist.delay(i, nrand) for i in range(b, bmax)]
      hsum = 0
      while len(procs) > 0:
         pending = []
         for proc in procs:
            if proc.ready():
               msg = proc.get()
               h = pickle.loads(base64.b64decode(msg))
               if hsum:
                  hsum.Add(h)
               else:
                  hsum = h
            else:
               pending.append(proc)
         print(f"{len(pending)} still pending in batch {b}")
         procs = pending
   return hsum
