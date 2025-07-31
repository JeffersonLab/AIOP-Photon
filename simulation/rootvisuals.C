//
// rootvisuals.C - ROOT macro to create visuals for the animation of
//                 the diamond and electron beam spot needed by the
//                 spotfinder web app.
//
// author: richard.t.jones at uconn.edu
// version: june 4, 2022
//

#include <iostream>
#include <TROOT.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TObjArray.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TStyle.h>
#include <TString.h>
#include <TExec.h>

#include "Couples.h"
#include "CobremsGeneration.hh"

TString resultsdir("/var/www/html/tools/spotfinder/tmp");

// These defaults can be changed from the spotfinder web tool
CobremsGeneration cobrems(12.0, 9.0);
struct CobremsInitializer {
   CobremsInitializer() {
      cobrems.setBeamErms(0.001);
      cobrems.setBeamEmittance(4.2e-9);
      cobrems.setCollimatorSpotrms(0.5e-3);
      cobrems.setCollimatorDistance(76.);
      cobrems.setCollimatorDiameter(3.4e-3);
      cobrems.setTargetCrystal("diamond");
      cobrems.setTargetThickness(50.e-6);
   }
} cobrems_init;

TObjArray *beamspot(double xsigma, double ysigma, double xycorr,
                    double resol=0.01, double width=20, double height=20)
{
   // Creates a 2D histogram representing the transverse current
   // density distribution of the electron bream as it passes through
   // the diamond radiator in the Hall D goniometer, modeled as a
   // general 2D ellipse. Distance units are mm.

   int nxbins(width/resol + 1);
   int nybins(height/resol + 1);
   TH2D *h2 = (TH2D*)gROOT->FindObject("beamspot");
   if (h2 != 0)
      delete h2;
   h2 = new TH2D("beamspot", "", nxbins, -width/2, +width/2,
                                 nybins, -height/2, +height/2);
   double vdeterm = xsigma*xsigma * ysigma*ysigma * (1 - xycorr*xycorr);
   vdeterm /= resol*resol;
   double covinv[3] = {ysigma*ysigma/vdeterm,
                      -xsigma*ysigma*xycorr/vdeterm,
                       xsigma*xsigma / vdeterm};
   int mx = 10*xsigma/resol;
   int my = 10*ysigma/resol;
   int midx = nxbins / 2;
   int midy = nybins / 2;
   for (int i = -mx; i < +mx; ++i) {
      for (int j = -my; j < +my; ++j) {
         double rho2 = i*i*covinv[0] + 2*i*j*covinv[1] + j*j*covinv[2];
         h2->SetBinContent(midx+i, midy+j, exp(-0.5*rho2));
      }
   }
   h2->SetMinimum(0.1);
   h2->SetStats(0);
   h2->GetXaxis()->SetTitle("beam x (mm)");
   h2->GetXaxis()->SetLabelSize(0.025);
   h2->GetXaxis()->SetTitleSize(0.025);
   h2->GetYaxis()->SetTitle("beam y (mm)");
   h2->GetYaxis()->SetLabelSize(0.025);
   h2->GetYaxis()->SetTitleSize(0.025);
   h2->GetYaxis()->SetTitleOffset(1.1);
   TObjArray *hset = new TObjArray();
   hset->Add(h2);
   return hset;
}

TObjArray *tiltspot(const char* radiator_name, int radiator_view,
                    double xoffset, double yoffset, double phideg,
                    double xsigma, double ysigma, double xycorr,
                    double resol=0.01, double width=20, double height=20)
{
   // Creates a 2D histogram representing the tilt distrubtion of the
   // diamond radiator identified by radiator_name, assuming a gaussian
   // 2D beam intensity profile with elliptical beam spot parameters
   // xsigma, ysigma, and xycorr. To locate this beam spot in radiator
   // coordinates, the crystal is shifted by xoffset,yoffset from the
   // nominal position with the beam centered on the radiator, and then
   // rotated about its center in the counterclockwise direction (beam
   // view) by angle phideg (degrees) relative to the orientation of
   // the crystal that is seen in the topograph Couples maps. These
   // active transformations on the radiator need to be converted into
   // their passive counterparts in order to find the beam spot in the
   // frame represented by the x and y axes in the topographs. The tilt
   // s a couple (thetaH, thetaV) that are sampled from the topographs
   // amed "hmu_moco" that are looked up using the Couples object for
   // the pair of scans selected by radiator_view as follows:
   //  radiator_view=0: getmap(1, "hmu_moco"), getmap(0, "hmu_moco")
   //  radiator_view=1: getmap(3, "hmu_moco"), getmap(2, "hmu_moco")
   //  radiator_view=2: getmap(1, "hmu_moco"), getmap(2, "hmu_moco")
   //  radiator_view=3: getmap(3, "hmu_moco"), getmap(0, "hmu_moco")
   // The result is returned in a 2D histogram of probability vs thetaH
   // on the x axis, thetaV on the y axis, centered on the tilt angle
   // in the center of the beam. Parameters resol, width, and height
   // determine the range and bin width of the output histogram. Units
   // for distance are mm, for angles are milliradians.

   // fetch the topographs from the Couples object
   int idx[4] = {1,3,1,3};
   int idy[4] = {0,2,2,0};
   if (radiator_view > 3)
      return 0;
   TFile fcouples((resultsdir + "/" + radiator_name + "_couples.root").Data());
   Couples *coup = (Couples*)gROOT->FindObject("coup");
   TH2D *h2topo[2] = {coup->getmap(idx[radiator_view], "hmu_moco"),
                      coup->getmap(idy[radiator_view], "hmu_moco")};
   double xrange[2] = {h2topo[0]->GetXaxis()->GetXmin(),
                       h2topo[0]->GetXaxis()->GetXmax()};
   double yrange[2] = {h2topo[0]->GetYaxis()->GetXmin(),
                       h2topo[0]->GetYaxis()->GetXmax()};
   double xorigin = (xrange[0] + xrange[1]) / 2;
   double yorigin = (yrange[0] + yrange[1]) / 2;

   // find the mean tilt angles over the entire crystal
   double topo_mr = 0.001; // conversion from topograph units to milliradians
   double mean_topo[2];
   for (int i=0; i < 2; ++i) {
      TH2D *h2 = h2topo[i];
      h2->SetDirectory(0);
      double sums[2] = {0,0};
      for (int j=1; j <= h2topo[i]->GetNcells(); ++j) {
         double t = h2->GetBinContent(j);
         if (t != 0) {
            sums[0]++;
            sums[1] += t;
         }
      }
      mean_topo[i] = sums[1] / (sums[0] + 1e-99);
      mean_topo[i] *= topo_mr;
   }

   // mark off the KEEP-OUT region of the mounting bar, lower right corner
   int xydim[2] = {h2topo[0]->GetNbinsX(), h2topo[0]->GetNbinsY()};
   int edge_margin = 10;
   int pledge[2][2] = {};
   for (int j=1; pledge[0][0] == 0; j++) {
      int i = xydim[0] - edge_margin - j;
      if (h2topo[0]->GetBinContent(i,j) > 0) {
         pledge[0][0] = i;
         pledge[0][1] = j;
      }
   }
   for (int j=edge_margin; pledge[1][0] == 0; j++) {
      int i = xydim[0] + edge_margin - j;
      if (h2topo[0]->GetBinContent(i,j) > 0) {
         pledge[1][0] = i;
         pledge[1][1] = j;
      }
   }
   double slope = (pledge[1][0] - pledge[0][0]) / 
                  (pledge[1][1] - pledge[0][1] + 1e-99);
   double aluminum = -999;
   int nfilled = 1;
   for (int j=1; nfilled > 0; j++) {
      nfilled = 0;
      int i = pledge[0][0] + (j - pledge[0][1]) * slope;
      for (; i <= xydim[0]; i++) {
         if (h2topo[0]->GetBinContent(i,j) == 0) {
            h2topo[0]->SetBinContent(i,j,aluminum);
            nfilled++;
         }
      }
   }

   // define the 2D tilt histogram
   int nbins[2] = {int(width / resol), int(height / resol)};
   double low_tilt[2] = {mean_topo[0] - width/2, mean_topo[1] - height/2};
   double high_tilt[2] = {mean_topo[0] + width/2, mean_topo[1] + height/2};
   TH2D *h2tilt = (TH2D*)gROOT->FindObject("h2tilt");
   if (h2tilt != 0)
      delete h2tilt;
   h2tilt = new TH2D("tiltspot", "2D tilt distribution",
                     nbins[0], low_tilt[0], high_tilt[0],
                     nbins[1], low_tilt[1], high_tilt[1]);
 
   // also save an image of the beam ellipse in the same frame
   TH2D *h2spot = (TH2D*)h2topo[0]->Clone("beamspot4topo");
   h2spot->SetTitle("beam spot overlay for topographs");
   h2topo[0]->SetName("topo4beamspot");
   h2topo[1]->SetName("topo4beamspot2");
   h2spot->Reset();

   // find the position and shape of the beam spot on the topograph maps
   double cosphi = cos(phideg * M_PI/180);
   double sinphi = sin(phideg * M_PI/180);
   double xcenter = xorigin - (xoffset*cosphi - yoffset*sinphi);
   double ycenter = yorigin - (yoffset*cosphi + xoffset*sinphi);
   double covar[3] = {xsigma*xsigma, xsigma*ysigma*xycorr, ysigma*ysigma};
   // determinant is an invariant under rotations
   double vdeterm = xsigma*xsigma * ysigma*ysigma * (1 - xycorr*xycorr);
   // rotate the covariance matrix, then take the inverse
   double R[2][2] = {{cosphi, sinphi},
                    {-sinphi, cosphi}};
   double covrot[3] = {R[0][0]*covar[0]*R[0][0] + R[1][0]*covar[1]*R[0][0] +
                       R[1][0]*covar[2]*R[1][0] + R[0][0]*covar[1]*R[1][0],
                       R[0][0]*covar[0]*R[0][1] + R[1][0]*covar[1]*R[0][1] +
                       R[1][0]*covar[2]*R[1][1] + R[0][0]*covar[1]*R[1][1],
                       R[0][1]*covar[0]*R[0][1] + R[1][1]*covar[1]*R[0][1] +
                       R[1][1]*covar[2]*R[1][1] + R[0][1]*covar[1]*R[1][1]};
   double toporesol = h2topo[0]->GetXaxis()->GetBinWidth(1);
   vdeterm /= toporesol*toporesol;
   double covinv[3] = {covrot[2]/vdeterm, -covrot[1]/vdeterm, covrot[0]/vdeterm};
   // scan the region of the beam spot, fill the tilt histogram
   int mx = 10*xsigma/toporesol;
   int my = 10*ysigma/toporesol;
   int midi = h2topo[0]->GetXaxis()->FindBin(xcenter);
   int midj = h2topo[0]->GetYaxis()->FindBin(ycenter);
   double intensity_sum[3] = {1e-99, 0, 0}; // all, crystal, aluminum
   for (int i = -mx; i < +mx; ++i) {
      double x = xcenter + i * toporesol;
      if (x <= xrange[0] || x >= xrange[1])
         continue;
      for (int j = -my; j < +my; ++j) {
         double y = ycenter + j * toporesol;
         if (y <= yrange[0] || y >= yrange[1])
            continue;
         double thetah = h2topo[0]->Interpolate(x,y) * topo_mr;
         double thetav = h2topo[1]->Interpolate(x,y) * topo_mr;
         double rho2 = i*i*covinv[0] + 2*i*j*covinv[1] + j*j*covinv[2];
         double intens = exp(-0.5*rho2);
         h2spot->SetBinContent(midi + i, midj + j, intens);
         intensity_sum[0] += intens;
         if (thetah > 0 && thetav > 0) {
            h2tilt->Fill(thetah, thetav, intens);
            intensity_sum[1] += intens;
         }
         else if (thetah < 0) {
            intensity_sum[2] += intens;
         }
      }
   }
   TString title;
   title.Form("beam fraction on crystal %lf, on mount %lf",
              intensity_sum[1] / intensity_sum[0], 
              intensity_sum[2] / intensity_sum[0]);
   h2topo[0]->SetTitle(title.Data());
   h2topo[1]->SetTitle(title.Data());
   h2spot->SetTitle(title.Data());
   h2tilt->SetTitle(title.Data());
   h2tilt->GetXaxis()->SetTitle("horizontal tilt #theta_{H} (mr)");
   h2tilt->GetYaxis()->SetTitle("vertical tilt #theta_{V} (mr)");
   h2tilt->GetYaxis()->SetTitleOffset(1.1);
   h2tilt->SetStats(0);
   h2tilt->SetDirectory(0);
   h2spot->SetDirectory(0);
   TObjArray *hset = new TObjArray();
   hset->Add(h2tilt);
   hset->Add(h2spot);
   hset->Add(h2topo[0]);
   hset->Add(h2topo[1]);
   for (int i=0; i < hset->GetEntries(); i++) {
      TH2D *h = (TH2D*)hset->At(i);
      h->GetXaxis()->SetLabelSize(0.025);
      h->GetXaxis()->SetTitleSize(0.025);
      h->GetYaxis()->SetLabelSize(0.025);
      h->GetYaxis()->SetTitleSize(0.025);
   }
   return hset;
}

TObjArray *cobrems_intensity(const char* radiator_name, int radiator_view,
                             double ebeam, double ibeam, double xyresol,
                             double thetah, double thetav,
                             double xoffset, double yoffset, double phideg,
                             double xsigma, double ysigma, double xycorr,
                             double eresol, double emin, double emax,
                             int polarized=0)
{
   // Creates a 1D histogram representing the collimated photon beam
   // intensity spectrum under the conditions described in the argument
   // list. Distance units are in mm, energies are in GeV, tilt angles
   // thetah, thetav are in milliradians, and phideg is in degrees.
   // Beam current ibeam is in microAmperes.

   // configure the coherent bremsstrahlung calculator
   cobrems.setBeamEnergy(ebeam);
   cobrems.setTargetOrientation(thetah * 1e-3, thetav * 1e-3, 0);
   cobrems.setPhotonEnergyMin(emin / 10.);
   cobrems.setCollimatedFlag(1);
   cobrems.setPolarizedFlag(polarized);

   // sample the intensity spectrum without smearing
   int nbins = ebeam / eresol;
   const char *names[2] = {"cobrems_intensity", "polar_intensity"};
   TProfile *hintens = (TProfile*)gROOT->FindObject(names[polarized]);
   if (hintens != 0)
      delete hintens;
   hintens = new TProfile(names[polarized], "",
                          nbins, 0, eresol * nbins, 0, 1.0e15);
   int i0 = hintens->GetXaxis()->FindBin(emin - 0.2);
   int i1 = hintens->GetXaxis()->FindBin(emax + 0.2);
   int ndiv = i1 - i0 + 1;
   double x[ndiv];
   double y[ndiv];
   for (int i=0; i<ndiv; i++) {
      x[i] = hintens->GetXaxis()->GetBinCenter(i0 + i) / ebeam;
      y[i] = cobrems.Rate_dNcdx(x[i]);
   }

   // apply beam-crystal smearing
   cobrems.applyBeamCrystalConvolution(ndiv, x, y);

   // convert to a histogram of beam intensity
   double norm = ibeam / 1.6e-13 / ebeam;
   for (int i=0; i<ndiv; i++) {
      hintens->Fill(x[i] * ebeam, y[i] * norm);
   }
   hintens->GetXaxis()->SetTitle("photon energy (GeV)");
   hintens->GetYaxis()->SetTitle("collimated beam rate (/s/GeV)");
   hintens->GetXaxis()->SetRangeUser(emin, emax);
   hintens->GetXaxis()->SetLabelSize(0.025);
   hintens->GetXaxis()->SetTitleSize(0.025);
   hintens->GetYaxis()->SetLabelSize(0.025);
   hintens->GetYaxis()->SetTitleSize(0.025);
   hintens->SetMinimum(0);
   hintens->SetStats(0);
   TObjArray *hset = new TObjArray();
   hset->Add(hintens);
   return hset;
}

int close_all_root_files()
{
   int count = 0;
   for (auto fileObj : *gROOT->GetListOfFiles()) {
      ((TFile*)fileObj)->Close();
      count++;
   }
   return count;
}

TObjArray *amorph_intensity(double ebeam, double ibeam,
                            double eresol, double emin, double emax)
{
   // Creates a 1D histogram representing the collimated photon beam
   // intensity spectrum for an amorphous radiator of the same thickness
   // as the dimaond.

   // configure the coherent bremsstrahlung calculator
   cobrems.setBeamEnergy(ebeam);
   cobrems.setTargetOrientation(175, 250, 0);
   cobrems.setPhotonEnergyMin(emin / 10.);
   cobrems.setCollimatedFlag(1);
   cobrems.setPolarizedFlag(0);

   // sample the intensity spectrum without smearing
   int nbins = ebeam / eresol;
   TProfile *hintens = (TProfile*)gROOT->FindObject("amorph_intensity");
   if (hintens != 0)
      delete hintens;
   hintens = new TProfile("amorph_intensity", "",
                          nbins, 0, eresol * nbins, 0, 1.0e15);
   int i0 = hintens->GetXaxis()->FindBin(emin - 0.2);
   int i1 = hintens->GetXaxis()->FindBin(emax + 0.2);
   int ndiv = i1 - i0 + 1;
   double norm = ibeam / 1.6e-13 / ebeam;
   for (int i=0; i<ndiv; i++) {
      double x = hintens->GetXaxis()->GetBinCenter(i0 + i) / ebeam;
      double y = cobrems.Rate_dNidx(x);
      hintens->Fill(x * ebeam, y * norm);
   }

   hintens->GetXaxis()->SetTitle("photon energy (GeV)");
   hintens->GetYaxis()->SetTitle("collimated beam rate (/s/GeV)");
   hintens->GetXaxis()->SetRangeUser(emin, emax);
   hintens->GetXaxis()->SetLabelSize(0.025);
   hintens->GetXaxis()->SetTitleSize(0.025);
   hintens->GetYaxis()->SetLabelSize(0.025);
   hintens->GetYaxis()->SetTitleSize(0.025);
   hintens->SetMinimum(0);
   hintens->SetStats(0);
   TObjArray *hset = new TObjArray();
   hset->Add(hintens);
   return hset;
}

void draw_beam_spot(TH2D *h2, int size_px, const char* outfile)
{
   gROOT->SetBatch(kTRUE);
   TCanvas *c1 = (TCanvas*)gROOT->FindObject("c1");
   if (c1 != 0)
      delete c1;
   c1 = new TCanvas("c1", "c1", 0, 0, size_px, size_px);
   gStyle->SetPalette(kInvertedDarkBodyRadiator,0,0.6);
   h2->Draw("COL");
   c1->SaveAs(outfile);
}

void draw_beam_tilt(TH2D *h2, int size_px, const char* outfile)
{
   gROOT->SetBatch(kTRUE);
   TCanvas *c1 = (TCanvas*)gROOT->FindObject("c1");
   if (c1 != 0)
      delete c1;
   c1 = new TCanvas("c1", "c1", 0, 0, size_px, size_px);
   gStyle->SetPalette(kBird,0,1.0);
   h2->Draw("COLZ");
   c1->SaveAs(outfile);
}

void draw_beamspot_on_topo(TH2D *h2spot, TH2D *h2topo, 
                           int size_px, const char* outfile)
{
   gROOT->SetBatch(kTRUE);
   TCanvas *c1 = (TCanvas*)gROOT->FindObject("c1");
   if (c1 != 0)
      delete c1;
   c1 = new TCanvas("c1", "c1", 0, 0, size_px, size_px);
   TPad *pad1 = new TPad("pad1","", 0,0,1,1);
   TPad *pad2 = new TPad("pad2","", 0,0,1,1);
   pad2->SetFillStyle(4000); // pad2 is transparent
   TExec *ex1 = new TExec("ex1", "gStyle->SetPalette(kBird, 0, 1.0);");
   TExec *ex2 = new TExec("ex2", "gStyle->SetPalette(kInvertedDarkBodyRadiator, 0, 0.6);");
   pad1->Draw();
   pad1->cd();
   ex1->Draw();
   h2topo->Draw("col");
   pad1->Update();
   c1->cd();
   double x1,y1,x2,y2;
   pad1->GetRange(x1,y1,x2,y2);
   pad2->Range(x1,y1,x2,y2);
   pad2->Draw();
   pad2->cd();
   ex2->Draw();
   h2spot->SetMinimum(0.1);
   h2spot->SetMaximum(1.0);
   h2spot->Draw("col same");
   pad2->Update();
   ex1->Draw();
   pad1->Update();
   c1->SaveAs(outfile);
}

void draw_cobrems_spectrum(TH1D *h1, int size_px, const char* outfile)
{
   gROOT->SetBatch(kTRUE);
   TCanvas *c1 = (TCanvas*)gROOT->FindObject("c1");
   if (c1 != 0)
      delete c1;
   c1 = new TCanvas("c1", "c1", 0, 0, size_px, size_px);
   h1->SetMinimum(0);
   h1->Draw();
   c1->SaveAs(outfile);
}
