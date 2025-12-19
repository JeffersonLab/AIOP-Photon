//
// Couples.C - implementation for Couples class.
//
// author: richard.t.jones at uconn.edu
// version: june 28, 2019
//
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Couples class                                                        //
//                                                                      //
// The Couples class provides a robust toolkit for relating pairs of    //
// rocking curve topographical maps within the root framework. It is    //
// based on the underlying Map2D class that extends the basic ROOT      //
// TH2D class with functionality related to treating 2D histograms as   //
// representations of 2D surfaces. Couples is a container for pairs of  //
// rocking curve topographs of the same Bragg reflection from the same  //
// sample taken along orthogonal directions, within which a projection  //
// of the crystal strain pattern across the entire crystal onto a 2D    //
// surface is visualized.                                               //
//                                                                      //
// The class currently supports the following on pairs of maps:         //
//        * association of complementary pairs of scans                 //
//        * arbitrary transformations (shifts, rotations etc.)          //
//        * masking of blank or random portions of the maps             //
//        * taking the curl, divergence in 2D                           //
//        * solving Poisson's equation in 2D                            //
//        * plotting pretty pictures                                    //
//                                                                      //
// This package was developed at the University of Connecticut by       //
// Richard T. Jones, as part of the analysis suite for the analysis     //
// of diamond crystal rocking curve scans at taken at CHESS and CLS,    //
// for the purpose of assessing coherent bremsstrahlung radiators for   //
// the GlueX experiment at Jefferson Lab.                               //
//                                                                      //
// This work is supported by the U.S. National Science Foundation       //
// under grant 1812415.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Couples.h"

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>

#include <TROOT.h>
#include <TCanvas.h>
#include <TExec.h>
#include <TStyle.h>
#include <TString.h>
#include <TRegexp.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>

using namespace std;

int nsearchpaths = 4;
std::string searchpath[] = 
{
   "/w/halld-scshelf2101/home/gjwei/AIOP/sept2025/AIOP-Photon/simulation/tmp",
   "/home/www/docs/halld/diamonds/spotfinder/tmp",
   "/home/www/docs/halld/diamonds/spotfinder/results",
   "root://nod29.phys.uconn.edu//Gluex/beamline/diamonds/cls-6-2019/results/",
   "root://nod29.phys.uconn.edu//Gluex/beamline/diamonds/cls-11-2017/results/"
};

Couples *Couples::fPicking = 0;

ClassImp(Couples);

Couples::Couples()
 : Couples("", "")
{
   // default constructor
}

Couples::Couples(const char *name, const char *title)
 : TNamed(name, title)
{
   // barebones constructor
 
   for (int p=0; p<4; p++) {
      fXshift[p] = 0;
      fYshift[p] = 0;
      fPsirot[p] = 0;
      fXscale[p] = 1;
      fYscale[p] = 1;
      fXYskew[p] = 0;
      fChirot[p] = 0;
      fPhirot[p] = 0;
      fXdisper[p] = 0;
      fYdisper[p] = 0;
      fYwalk[p] = 0;
      fCleanMask[p] = 0;
      fMap[p] = 0;
      fMmu[p] = 0;
   }
   fZeroMask = 0;
   fXrange[0] = 0;
   fXrange[1] = 8.5;
   fPixels[0] = 850;
   fYrange[0] = 0;
   fYrange[1] = 8.5;
   fPixels[1] = 850;
}

Couples::Couples(const Couples &src)
 : TNamed(src.GetName(), src.GetTitle())
{
   // copy constructor
 
   copy(src);
}

Couples::~Couples()
{
   // destructor
}

Couples &Couples::operator=(const Couples &src)
{
   // assignment operator

   copy(src);
   return *this;
}

void Couples::copy(const Couples &src)
{
   // safe copy of state from src to *this

   if (src.fZeroMask) {
      fZeroMask = new Map2D(*src.fZeroMask);
      fZeroMask->SetDirectory(0);
   }
   for (int p=0; p<4; p++) {
      fResultsPath[p] = src.fResultsPath[p];
      fResultsName[p] = src.fResultsName[p];
      fXshift[p] = src.fXshift[p];
      fYshift[p] = src.fYshift[p];
      fPsirot[p] = src.fPsirot[p];
      fXscale[p] = src.fXscale[p];
      fYscale[p] = src.fYscale[p];
      fXYskew[p] = src.fXYskew[p];
      fChirot[p] = src.fChirot[p];
      fPhirot[p] = src.fPhirot[p];
      fXdisper[p] = src.fXdisper[p];
      fYdisper[p] = src.fYdisper[p];
      fYwalk[p] = src.fYwalk[p];
      fMmu[p] = src.fMmu[p];
      if (src.fCleanMask[p]) {
         fCleanMask[p] = new Map2D(*src.fCleanMask[p]);
         fCleanMask[p]->SetDirectory(0);
      }
      else {
         fCleanMask[p] = 0;
      }
      if (src.fMap[p]) {
         fMap[p] = new Map2D(*src.fMap[p]);
         fMap[p]->SetDirectory(0);
      }
      else {
         fMap[p] = 0;
      }
   }
   fXrange[0] = src.fXrange[0];
   fXrange[1] = src.fXrange[1];
   fYrange[0] = src.fYrange[0];
   fYrange[1] = src.fYrange[1];
   fPixels[0] = src.fPixels[0];
   fPixels[1] = src.fPixels[1];
}

void Couples::select(int p, const char *resultsfile)
{
   // Select a scan results file (XXX_results.root) for either
   // the first (p=0) or second (p=1) member of the pair.

   TFile *fresults = open(resultsfile);
   if (fresults) {
      fResultsPath[p] = resultsfile;
      deletemap(p);
      if (fCleanMask[p]) {
         delete fCleanMask[p];
         fCleanMask[p] = 0;
      }
   }
}

void Couples::setresol(int xpixels, int ypixels)
{
   // Set the resolution of the images to be generated,
   // in terms of the width (xpixels) and height (ypixels) in pixels.

   if (ypixels == 0)
      ypixels = xpixels;
   fPixels[0] = xpixels;
   fPixels[1] = ypixels;
   for (int p=0; p<4; ++p)
      deletemap(p);
}

void Couples::setxrange(double xmax, double xmin)
{
   // Set the x coordinate range of the images to be generated,
   // in mm.
 
   fXrange[0] = xmin;
   fXrange[1] = xmax;
   for (int p=0; p<4; ++p)
      deletemap(p);
}

void Couples::setyrange(double ymax, double ymin)
{
   // Set the y coordinate range of the images to be generated,
   // in mm.

   fYrange[0] = ymin;
   fYrange[1] = ymax;
   for (int p=0; p<4; ++p)
      deletemap(p);
}

void Couples::shift(int p, double xshift, double yshift)
{
   // Set the offset in x (xshift) and y (yshift) of the images to be generated
   // for member p=0 or p=1, relative to the original topographs, in units of mm.

   fXshift[p] = xshift;
   fYshift[p] = yshift;
   deletemap(p);
}

void Couples::invert(int p, int ix, int iy)
{
   // Request that the generated images for member p=0 or p=1 be inverted
   // in x (ix != 0) and/or y (iy != 0) relative to how they appear in the
   // original topographs.

   if (ix) {
      fXscale[p] = -fabs(fXscale[p]);
      fXYskew[p] = -fabs(fXYskew[p]);
   }
   else {
      fXscale[p] = fabs(fXscale[p]);
      fXYskew[p] = fabs(fXYskew[p]);
   }
   if (iy) {
      fYscale[p] = -fabs(fYscale[p]);
      fXYskew[p] *= -1;
   }
   else
      fYscale[p] = fabs(fYscale[p]);
   deletemap(p);
}

void Couples::rotate(int p, double phi, int deg)
{
   // Request that the generated images for member p=0 or p=1 be rotated in
   // the xy plane by angle phi (radians) relative to how they appear in the
   // original topographs.
 
   if (deg)
      phi *= M_PI/180;
   fPsirot[p] = fmod(phi, 2*M_PI);
   deletemap(p);
}

void Couples::zoom(int p, double fzoom)
{
   // Request that the generated images for member p=0 or p=1 be zoomed by
   // factor fzoom the xy plane relative to how they appear in the
   // original topographs.
 
   fXscale[p] *= fzoom;
   fYscale[p] *= fzoom;
   deletemap(p);
}

void Couples::setmin(int p, double min)
{
   // Set the minimum of the colormap for topograph p.
   // This value affects the currently displayed image,
   // if any, but is not saved permanently.

   if (p < 0) {
      for (p=0; p<4; ++p)
         setmin(p, min);
   }
   else if (fMap[p])
      fMap[p]->SetMinimum(min);
}

void Couples::fillvoids(int p)
{
   // Call fillvoids on the colormap for topograph p.
   // This value affects the currently displayed image,
   // if any, but is not saved permanently.

   if (p < 0) {
      for (p=0; p<4; ++p)
         fillvoids(p);
   }
   else if (fMap[p]) {
      fMap[p]->FillVoids();
      if (fZeroMask) {
         Map2D mask(fMap[p]);
         mask.Reset();
         mask.Fill(fZeroMask);
         fMap[p]->Mask(&mask);
      }
   }
}

void Couples::setmax(int p, double max)
{
   // Set the minimum of the colormap for topograph p.
   // This value affects the currently displayed image,
   // if any, but is not saved permanently.

   if (fMap[p])
      fMap[p]->SetMaximum(max);
}

void Couples::stretch(int p, double aspectratio)
{
   // Request that the generated images for member p=0 or p=1 be stretched
   // by aspect ratio aspectratio (width / height) relative to how they 
   // appear in the original topographs.
 
   double scalefactor = sqrt(fabs(aspectratio));
   fXscale[p] = (fXscale[p] > 0)? scalefactor : -scalefactor;
   fYscale[p] = (fYscale[p] > 0)? 1/scalefactor : -1/scalefactor;
   fXYskew[p] = 0;
   deletemap(p);
}

void Couples::setmask(const Map2D *mask)
{
   // Request that all zero pixels in the Map2D object mask be zeroed in
   // the generated images for either of the two members of this set.
   // The caller retains ownership of the passed object.
 
   if (fZeroMask) {
      delete fZeroMask;
      fZeroMask = 0;
   }
   if (mask == 0)
      return;
   TH2D hmap("hmask", "mask image", fPixels[0], fXrange[0], fXrange[1],
                                    fPixels[1], fXrange[0], fXrange[1]);
   fZeroMask = new Map2D(hmap);
   fZeroMask->SetDirectory(0);
   fZeroMask->Fill(mask);
   for (int p=0; p<4; ++p)
      deletemap(p);
}

void Couples::setclean(int p, Map2D *mask)
{
   // Specify a clean-up masking operation to be applied to the raw 
   // images before any transformations are applied. Separate masks
   // are kept for the two members of the Couples. The caller retains
   // ownership of the passed mask object.
   //
   // Note that the CleanMask is different from ZeroMask. ZeroMask
   // gets applied after the transformations take place, and matches
   // the final maps in pixel dimensions and axis extents. Just one
   // instance of ZeroMask is the kept for both members.
 
   if (fCleanMask[p]) {
      delete fCleanMask[p];
      fCleanMask[p] = 0;
      deletemap(p);
   }
   if (mask == 0)
      return;
   TFile *fresults = open(fResultsPath[p]);
   if (fresults == 0) {
      printf("Error in Couples::setclean - no raw images found!\n");
      return;
   }
   TH2D *h2d = (TH2D*)fresults->Get("hmu");
   if (h2d == 0) {
      printf("Error in Couples::setclean - no hmu image found!\n");
      return;
   }
   fCleanMask[p] = new Map2D(*h2d);
   fCleanMask[p]->SetDirectory(0);
   fCleanMask[p]->Reset();
   fCleanMask[p]->Fill(mask);
   deletemap(p);
}

Map2D *Couples::getmask()
{
   // Return a copy of the zero mask, if any,
   // rescaled to the current dimensions and
   // resolution. The caller owns the returned object.

   if (fZeroMask)
      return reshape(fZeroMask);
   else
      return 0; 
}

Map2D *Couples::reshape(const Map2D *src) const
{
   // Return a copy of the input src map in a new Map2D
   // object whose dimensions and resolution match the 
   // current state of *this. Caller owns the returned object.

   TH2D h2("xyz","",fPixels[0],fXrange[0],fXrange[1],
                    fPixels[1],fYrange[0],fYrange[1]);
   Map2D *map = new Map2D(h2);
   map->SetName(src->GetName());
   map->SetTitle(src->GetTitle());
   map->Fill(src);
   return map;
}

Map2D *Couples::getclean(int p)
{
   // Return a copy of the clean mask, if any, for member p.
   // The caller owns the returned object.

   if (fCleanMask[p])
      return new Map2D(*fCleanMask[p]);
   else
      return 0; 
}

Map2D *Couples::getmap(int p, const char *name)
{
   // Return a copy of the named image from member p of this Couples,
   // if any.  If argument name is defaulted, whatever name was last
   // requested is returned. The caller owns the returned object.

   if (name == 0) {
      if (fMap[p] == 0 && fResultsName[p].Length() == 0) {
         printf("Error in Couples::getmap - map name is missing!\n");
         return 0;
      }
      name = fResultsName[p].Data();
   }
   else if (fResultsName[p] != name) {
      deletemap(p);
   }
   if (fMap[p] == 0) {
      TFile *fresults = open(fResultsPath[p]);
      if (fresults == 0)
         return 0;
      TH2D *h2d = (TH2D*)fresults->Get(name);
      if (h2d == 0)
         return 0;
      Map2D m2draw(h2d);
      if (fCleanMask[p])
         m2draw.Mask(fCleanMask[p]);
      if (fYwalk[p] != 0) {
         Map2D *mmu = getmmu(p);
         remove_walk(&m2draw, mmu, fYwalk[p]*1e-3);
      }
      double xmin = h2d->GetXaxis()->GetXmin();
      double ymin = h2d->GetYaxis()->GetXmin();
      double xmax = h2d->GetXaxis()->GetXmax();
      double ymax = h2d->GetYaxis()->GetXmax();
      double x0 = (xmax+xmin)/2;
      double y0 = (ymax+ymin)/2;
      xmin = x0 + (xmin - x0)*1.4;
      xmax = x0 + (xmax - x0)*1.4;
      ymin = y0 + (ymin - y0)*1.4;
      ymax = y0 + (ymax - y0)*1.4;
      Map2D &m2d = m2draw.Resize(xmin,ymin,xmax,ymax);
      m2d.SetName(m2draw.GetName());
      fResultsName[p] = name;
      if (fabs(fXYskew[p]) > 0.01) {
         printf("Warning in Couples::getmap - plotting with skew "
                "is not currently supported, ignoring x/y skew=%lf\n",
                fXYskew[p]);
      }
      m2d.Rescale(fXscale[p], fYscale[p], 1);
      m2d.Rotate(0, 0, fPsirot[p]);
      m2d.Shift(fXshift[p], fYshift[p], 0);
      if (fZeroMask) {
         Map2D mask(m2d);
         mask.Reset();
         mask.Fill(fZeroMask);
         m2d.Mask(&mask);
      }
      if (fResultsName[p](0,3) == "hmu") {
         m2d.Tilt(fXdisper[p],fYdisper[p]);
      }

      TH2D hmap("hmap", "", fPixels[0], fXrange[0], fXrange[1],
                            fPixels[1], fYrange[0], fYrange[1]);
      fMap[p] = new Map2D(hmap);
      fMap[p]->SetDirectory(0);
      fMap[p]->SetName(m2d.GetName());
      fMap[p]->Fill(&m2d);
      fMap[p]->Despeckle(fPixels[0]*fPixels[1]*0.4);
      fMap[p]->GetXaxis()->SetTitle("x (mm)");
      fMap[p]->GetYaxis()->SetTitle("y (mm)");
      fMap[p]->GetYaxis()->SetTitleOffset(1.2);
      fMap[p]->GetZaxis()->SetTitle("#murad");
      fMap[p]->GetZaxis()->SetTitleOffset(1.5);
      fMap[p]->SetContour(100);
      TString title(m2d.GetTitle());
      TRegexp basename("[^/]*$");
      fMap[p]->SetTitle(title + " from " + fResultsPath[p](basename));
   }
   return new Map2D(*fMap[p]);
}

Double_t Couples::getchi(int p) const
{
   return fChirot[p];
}

void Couples::setchi(int p, double chi, int deg)
{
   fChirot[p] = (deg)? chi * M_PI/180 : chi;
}

Double_t Couples::getphi(int p) const
{
   return fPhirot[p];
}

void Couples::setphi(int p, double phi, int deg)
{
   fPhirot[p] = (deg)? phi * M_PI/180 : phi;
}

int Couples::getdispersion(int p, double &xslope, double &yslope) const
{
   xslope = fXdisper[p];
   yslope = fYdisper[p];
   if (xslope != 0 || yslope != 0)
      return 1;
   else
      return 0;
}

void Couples::setdispersion(int p, double xslope, double yslope)
{
   fXdisper[p] = xslope;
   fYdisper[p] = yslope;
   deletemap(p);
}

Double_t Couples::getywalk(int p) const
{
   return fYwalk[p];
}

void Couples::setywalk(int p, double dydtheta)
{
   fYwalk[p] = dydtheta;
   deletemap(p);
}

void Couples::lineup(int p0, int p1)
{
   // Performs a dynamic visual alignment image p1 relative to image p0
   // in the couples, adjusting the internal parameters to keep track of
   // the net transformation leading to the final alignment. It works
   // using the latest generated images from each of the two members.
   // Typically one performs a draw(), followed by a rough set of shifts
   // + rotations + inversions to get the two images close to alignment,
   // then finishes with a call to lineup().
 
   if (fMap[p0] == 0 || fMap[p1] == 0) {
      printf("Error in Couples::lineup - please update the two graphics "
             "displays with images you would like to align, "
             "then try again.\n");
      return;
   }
   printf(
   "Performing a dynamic visual alignment of two images. At the prompt,\n"
   "you can enter the following commands to tweak the alignment, where\n"
   "the factor M is an optional multiplier of the minimum change step:\n"
   " translation or rotation:\n"
   " [M]r : shift image 2 to the right\n"
   " [M]l : shift image 2 to the left\n"
   " [M]u : shift image 2 up\n"
   " [M]d : shift image 2 down\n"
   " [M]p : rotate image 2 counter-clockwise\n"
   " [M]n : rotate image 2 clockwise\n"
   " [M]i : zoom in (expand) image 2\n"
   " [M]o : zoom out (contract) image 2\n"
   " [M]s : expand x, contract y on image 2\n"
   " [M]t : contract x, expand y on image 2\n"
   " [M]b : make image 2 more bright\n"
   " [M]f : make image 2 more faint\n"
   " [M]B : make image 1 more bright\n"
   " [M]F : make image 1 more faint\n"
   " [M]q : quit\n"
   "    * : (or anything else) flip between the two images again\n"
   );
   TExec *pal1 = new TExec("pal1", "Couples::select_palette(0);");
   TExec *pal2 = new TExec("pal2", "Couples::select_palette(1);");
   while (true) {
      TCanvas *c1 = select_canvas(0);
      pal1->Draw();
      fMap[p0]->Draw("colz");
      c1->Update();
      usleep(5e5);
      pal2->Draw();
      fMap[p1]->Draw("col same");
      c1->Update();
      pal1->Draw();
      printf("r/l/u/d/p/n/i/o/s/t/b/f/B/F/q/*? ");
      std::cout << std::flush;
      std::string res;
      std::getline(std::cin, res);
      if (res.size() == 0)
         continue;
      char cmd = res.back();
      res.pop_back();
      int mult = 1;
      if (res.size() > 0)
         mult = std::atoi(res.c_str());
      if (cmd == 'r') {
         double dx = mult * fMap[p1]->GetXaxis()->GetBinWidth(1);
         fMap[p1]->Shift(dx, 0, 0);
         fXshift[p1] += dx;
      }
      else if (cmd == 'l') {
         double dx = mult * fMap[p1]->GetXaxis()->GetBinWidth(1);
         fMap[p1]->Shift(-dx, 0, 0);
         fXshift[p1] -= dx;
      }
      else if (cmd == 'u') {
         double dy = mult * fMap[p1]->GetYaxis()->GetBinWidth(1);
         fMap[p1]->Shift(0, dy, 0);
         fYshift[p1] += dy;
      }
      else if (cmd == 'd') {
         double dy = mult * fMap[p1]->GetYaxis()->GetBinWidth(1);
         fMap[p1]->Shift(0, -dy, 0);
         fYshift[p1] -= dy;
      }
      else if (cmd == 'p') {
         double dphi = mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rotate(0, 0, dphi);
         fPsirot[p1] += dphi;
         double sx = fXshift[p1];
         double sy = fYshift[p1];
         fXshift[p1] = sx * cos(dphi) - sy * sin(dphi);
         fYshift[p1] = sy * cos(dphi) + sx * sin(dphi);
      }
      else if (cmd == 'm') {
         double dphi = mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rotate(0, 0, -dphi);
         fPsirot[p1] -= dphi;
         double sx = fXshift[p1];
         double sy = fYshift[p1];
         fXshift[p1] = sx * cos(dphi) + sy * sin(dphi);
         fYshift[p1] = sy * cos(dphi) - sx * sin(dphi);
      }
      else if (cmd == 'i') {
         double dsize = 1 + mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rescale(dsize, dsize, 1);
         fXscale[p1] *= dsize;
         fYscale[p1] *= dsize;
      }
      else if (cmd == 'o') {
         double dsize = 1 - mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rescale(dsize, dsize, 1);
         fXscale[p1] *= dsize;
         fYscale[p1] *= dsize;
      }

// A linear coordinate transformation is used to map from from raw images
// from the camera to the maps that are returned by the getmap method.
//
//    r' = S + R I r
//
// where R and I are 2x2 matrices and S is a 2-vector.
//
//      / cos_phi  -sin_phi \
//  R = |                     |
//      \ sin_phi   cos_phi /
//
//       / Ix     Ixy \
//  I = |              |
//       \ 0      Iy  /
//
//  S = [ Sx  Sy ]
//
// The task in this code section is to update the current R, I, and S
// in such a way as to accumulate many small shifts, rotations, etc.
// to describe their net effect in a single final set R, I, and S.
// For displacement and rotation tweaks this is straightforward, as
// coded above.
//
// shifts dS: Sx -> Sx + dSx, Sy -> Sy + dSy
//
// tilts d_phi: phi -> phi + d_phi, 
//              Sx -> Sx cos_dphi - Sy sin_dphi
//              Sy -> Sy cos_dphi + Sx sin_dphi
//
// But for tweaks that affect the strain matrix I (stretch/squeeze), some work 
// is needed to see how these aggregate together into accumulated transform
// matrices. Consider a stretch by factor s along x, 1/s along y described by
//
//      / s   0  \
// D = |          |
//      \ 0  1/s /
//
// The task is to see how (S + D R I) gets rewritten as (S + R' I') such that
// the effect of D is incorporated into updates R -> R' and I -> I', further
// stipulating that I' must remain upper-diagonal as required in a minimal
// description of a general linear coordinate transform. The solution to this
// linear algebra problem is:
//
// R' = R At, where At is the transpose of an orthogonal matrix A, and
// I' = A Rt D R I
//
// where the new matrix A is introduced for the sole purpose of returning I'
// to upper-diagonal form.
//
//      / 1 -a \
// A = |        |
//      \ a  1 /
// 
// The following lines work out the solution for I' and R' to leading order
// in the small tweak parameter s.
//
//           / s cos_phi^2 + 1/s sin_phi^2      cos_phi sin_phi (1/s - s) \
// Rt D R = |                                                              |
//           \ cos_phi sin_phi (1/s - s)   s sin_phi^2 + (1/s) cos_phi^2  /
//
//                                       / cos_2phi  -sin_2phi \
//        = 1/2 [(s + 1/s) + (s - 1/s)  |                       | ]
//                                       \-sin_2phi  -cos_2phi /
//
//                        / cos_2phi  -sin_2phi \
//        ~= 1 + (s - 1) |                       | 
//                        \-sin_2phi  -cos_2phi /
//
//                          / Ix cos_2phi      Ixy cos_2phi - Iy sin_2phi \
// Rt D R I ~= I + (s - 1) |                                               |
//                          \ -Ix sin_2phi    -Ixy sin_2phi - Iy cos_2phi /
//
// This result is not ready to be adopted as I' because it is not in upper-
// diagonal form yet. For That, we need to find the right matrix A that can
// make this happen. Consider the general form,
//
//      / 1  -a \   / u   v \      / u - a w   v - a x \
//     |         | |         |  = |                     |
//      \ a   1 /   \ w   x /      \ w + a u   x + a v /
//
// To make the rhs upper-diagonal, we need to let
//
//  a = -w/u = (s - 1) sin_2phi / [1 + (s - 1) cos_2phi]
//   ~= (s - 1) sin_2phi 
//
// to leading order in the small parameter s-1. Putting all of this together,
//
//                    / Ix cos_2phi   Ixy cos_2phi - 2 Iy sin_2phi \
// I' ~= I + (s - 1) |                                              |
//                    \    0                          -Iy cos_2phi /
//
//       / cos_phi  -sin_phi \   / 1 -a \      / cos_phi'   -sin_phi' \
// R' = |                     | |        |  = |                        |
//       \ sin_phi   cos_phi /   \ a  1 /      \ sin_phi'    cos_phi' /
//
// where phi' = phi + (s - 1) sin_2phi, and
//
// S' = [s Sx, (1/s) Sy ]

      else if (cmd == 's') {
         double dsize = 1 + mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rescale(dsize, 1/dsize, 1);
         fXshift[p1] *= dsize;
         fYshift[p1] /= dsize;
         double cos2phi = cos(2 * fPsirot[p1]);
         double sin2phi = sin(2 * fPsirot[p1]);
         fXscale[p1] *= 1 + (dsize - 1) * cos2phi;
         fYscale[p1] *= 1 - (dsize - 1) * cos2phi;
         fXYskew[p1] *= 1 + (dsize - 1) * cos2phi;
         fXYskew[p1] -= 2 * (dsize - 1) * sin2phi * fYscale[p1];
         fPsirot[p1] += (dsize - 1) * sin2phi;
      }
      else if (cmd == 't') {
         double dsize = 1 - mult * 2.0 / fMap[p1]->GetNbinsX();
         fMap[p1]->Rescale(dsize, 1/dsize, 1);
         fXshift[p1] *= dsize;
         fYshift[p1] /= dsize;
         double cos2phi = cos(2 * fPsirot[p1]);
         double sin2phi = sin(2 * fPsirot[p1]);
         fXscale[p1] *= 1 + (dsize - 1) * cos2phi;
         fYscale[p1] *= 1 - (dsize - 1) * cos2phi;
         fXYskew[p1] *= 1 + (dsize - 1) * cos2phi;
         fXYskew[p1] -= 2 * (dsize - 1) * sin2phi * fYscale[p1];
         fPsirot[p1] += (dsize - 1) * sin2phi;
      }
      else if (cmd == 'b') {
         double zmax = fMap[p1]->GetMaximum();
         fMap[p1]->SetMaximum(zmax * (1 - 0.01 * mult));
      }
      else if (cmd == 'f') {
         double zmax = fMap[p1]->GetMaximum();
         fMap[p1]->SetMaximum(zmax * (1 + 0.01 * mult));
      }
      else if (cmd == 'B') {
         double zmax = fMap[p0]->GetMaximum();
         fMap[p0]->SetMaximum(zmax * (1 - 0.01 * mult));
      }
      else if (cmd == 'F') {
         double zmax = fMap[p0]->GetMaximum();
         fMap[p0]->SetMaximum(zmax * (1 + 0.01 * mult));
      }
      else if (cmd == 'q') {
         break;
      }
   }
}

void Couples::select_palette(int pal)
{
   if (pal == 0)
      gStyle->SetPalette(kBird, 0, 1.0);
   else
      gStyle->SetPalette(kBird, 0, 0.5);
}

Map2D *Couples::cleanup(int p, int minspeck)
{
   // Examine the pixels of the rocking curve topographs from the first
   // member (p=0) or second member (p=1) of this couple, and distinguish
   // the ones that show a legitimate diffraction peak from those that do
   // not. Zero any pixels that do not have a legitimate diffraction peak
   // and return a pointer to the image as a Map2D. The caller becomes
   // the owner of the returned object.
   //
   // This algorithm has been tested on the images taken during run
   // cls-6-2019. It may need to be tweaked to work properly on images
   // taken from other runs.

   TFile *fresults = open(fResultsPath[p]);
   if (fresults == 0)
      return 0;
   TH2D *hbase = (TH2D*)fresults->Get("hbase");
   if (hbase == 0)
      return 0;
   TH2D *hamp = (TH2D*)fresults->Get("hamp");
   if (hamp == 0)
      return 0;
   hamp->Add(hbase);
   int col1 = 1;
   int colN = hamp->GetNbinsX();
   int row1 = 1;
   int rowN = hamp->GetNbinsY();
   TH1D *hedge[4];
   hedge[0] = hamp->ProjectionX("col1", 1, rowN);
   if (hedge[0]->Integral() == 0) {
      printf("Error in Couples::cleanup - map is empty!\n");
      return 0;
   }
   while (hedge[0]->GetBinContent(col1) == 0)
      ++col1;
   while (hedge[0]->GetBinContent(colN) == 0)
      --colN;
   hedge[1] = hamp->ProjectionY("row1", 1, colN);
   while (hedge[1]->GetBinContent(row1) == 0)
      ++row1;
   while (hedge[1]->GetBinContent(rowN) == 0)
      --rowN;
   hedge[0] = hamp->ProjectionX("row1", row1+1, row1+1);
   hedge[1] = hamp->ProjectionY("col1", col1+1, col1+1);
   hedge[2] = hamp->ProjectionX("rowN", rowN-1, rowN-1);
   hedge[3] = hamp->ProjectionY("colN", colN-1, colN-1);
   double bgheight = 9999;
   for (int side=0; side<4; ++side) {
      double height = hedge[side]->Integral() / hedge[side]->GetNbinsX();
      if (height < bgheight)
         bgheight = height;
      delete hedge[side];
   }
   Map2D *silh = new Map2D(*hamp);
   Map2D silh0(*silh);
   int speck0 = silh0.Despeckle();
   silh->ZeroSuppress(bgheight);
   Map2D silh1(*silh);
   int speck1 = silh1.Despeckle();
   double bgheight_step[2];
   if (bgheight > 100) {
      bgheight_step[0] = 1;
      bgheight_step[1] = 0.1;
   }
   else {
      bgheight_step[0] = 0.1;
      bgheight_step[1] = 0.05;
   }
   while (speck1 < speck0 * 1.2) {
      bgheight += bgheight_step[0];
      silh->ZeroSuppress(bgheight);
      Map2D silh2(*silh);
      speck1 = silh2.Despeckle();
   }
   int speck2 = speck1;
   while (speck2 >= speck1 || speck2 < minspeck) {
      bgheight += bgheight_step[1];
      silh->ZeroSuppress(bgheight);
      speck1 = speck2;
      Map2D silh3(*silh);
      speck2 = silh3.Despeckle();
   }
   silh->Despeckle(10000);
   silh->SetDirectory(0);
   select_canvas(p);
   silh->Draw("colz");
   setclean(p, silh);
   deletemap(p);
   return silh;
}

void Couples::polycrop(int p, const char *name)
{
   // Display the topograph image with the given name belonging
   // to the first (p=0) or second (p=1) member of this couple
   // as a topological color map and ask the user to select the
   // vertices of a polygon to use for cropping the extents of
   // the image. All pixels outside the selected polygon are set
   // to matte (zero).
   //
   // WARNING: Only the copy of the named map in memory is modified
   // by this method. If you want to use the adopt the result to
   // update the mask for this couple, take the object returned
   // by getmap(p,name) after this call, mask it with *getmask()
   // if any, and register the result using setmask().

   if (name == 0 && fMap[p] == 0) {
      printf("Error in Couples::polycrop - no image found!\n");
      return;
   }
   else if (getmap(p, name) == 0) {
      printf("Error in Couples::polycrop - image not available!\n");
      return;
   }
   TCanvas *can = select_canvas(p);
   fPolyVertex.clear();
   can->DeleteExec("polypicker");
   if (p == 0)
      can->AddExec("polypicker", "Couples::polypicker(0);");
   else if (p == 1)
      can->AddExec("polypicker", "Couples::polypicker(1);");
   else if (p == 2)
      can->AddExec("polypicker", "Couples::polypicker(2);");
   else if (p == 3)
      can->AddExec("polypicker", "Couples::polypicker(3);");
   else  {
      printf("Error in Couples::polycrop - invalid argument %d\n", p);
      return;
   }
   fMap[p]->SetStats(0);
   fMap[p]->Draw("colz");
   can->Update();
   printf("Use the mouse to pick the vertices of the cropping polygon\n"
          "in window c%d, then click the middle mouse button to terminate.\n"
          "The polygon will close itself by connecting the last vertex\n"
          "to the first.\n", p+1);
   std::cout << "waiting..." << std::flush;
   fPicking = this;
}

void Couples::polypicker(int p)
{
   // This callback is called whenever a mouse event (move, click, etc.)
   // happens inside one of the open ROOT graphics windows. This one
   // activates when the user left-clicks inside the active region of a
   // displayed TH2D histogram any number of times to select a polygonal
   // region of interest, then right-clicks in the image to execute the
   // cropping action.

   TCanvas *can;
   if (p == 0)
      can = (TCanvas*)gROOT->FindObject("c1");
   else if (p == 1)
      can = (TCanvas*)gROOT->FindObject("c2");
   else if (p == 2)
      can = (TCanvas*)gROOT->FindObject("c3");
   else
      can = (TCanvas*)gROOT->FindObject("c4");
   TObject *select = can->GetSelected();
   if (!select) {
      //printf("none selected!\n");
      return;
   }
   if (!select->InheritsFrom("Map2D")) {
      //printf("not a Map2D object!\n");
      return;
   }
   Map2D *refmap = (Map2D*)select;
   int event = can->GetEvent();
   if (event == 1) {
      double x = can->AbsPixeltoX(can->GetEventX());
      double y = can->AbsPixeltoY(can->GetEventY());
      int px = refmap->GetXaxis()->FindBin(x);
      int py = refmap->GetYaxis()->FindBin(y);
      fPicking->fPolyVertex.push_back(std::pair<int,int>(px,py));
      printf("cropping polygon vertex %ld (%d,%d)\n",
             fPicking->fPolyVertex.size(), px, py);
   }
   else if (event == 12) {
      std::cout << "polygon closed, cropping..." << std::flush;
      can->DeleteExec("polypicker");
      fPicking->polycropper(p);
      std::cout << "done." << std::endl;
      fPicking = 0;
   }
}

void Couples::polycropper(int p)
{
   // Clear all pixels in the topo map that lie outside the cropping region
   // defined as the interior of a polygon with vertices stored in vector
   // fPolyVertex, that have just been selected by the user with the mouse.
   // This algorithm assumes that the sides of the polygon are touching only
   // at the vertices.

   TCanvas *can;
   if (p == 0)
      can = (TCanvas*)gROOT->FindObject("c1");
   else if (p == 1)
      can = (TCanvas*)gROOT->FindObject("c2");
   else if (p == 2)
      can = (TCanvas*)gROOT->FindObject("c3");
   else
      can = (TCanvas*)gROOT->FindObject("c4");
   TObject *select = can->GetSelected();
   TH2D *refmap = (TH2D*)select;
   Map2D mask(*refmap);
   mask.Flood(1);
   int nv = fPolyVertex.size();
   int nborder = 0;
   for (int iv=0; iv<nv; ++iv) {
      int i = fPolyVertex[iv].first;
      int j = fPolyVertex[iv].second;
      mask.SetBinContent(i,j,1);
      int ie = fPolyVertex[(iv+1)%nv].first;
      int je = fPolyVertex[(iv+1)%nv].second;
      mask.SetBinContent(ie,je,0);
      nborder += 1;
      int dx = (ie > i)? +1 : -1;
      int dy = (je > j)? +1 : -1;
      double x = (i + 0.5) * dx;
      double y = (j + 0.5) * dy;
      double slope = abs((je - j)/(ie - i + 1e-9));
      while (mask.GetBinContent(i,j) > 0) {
         mask.SetBinContent(i,j,0);
         nborder += 1;
         double xb = floor(x + 1) - x;
         double yb = floor(y + 1) - y;
         if (yb > xb * slope) {
            x += xb;
            y += xb * slope;
            i += dx;
         }
         else if (yb < xb * slope) {
            x += yb / slope;
            y += yb;
            j += dy;
         }
         else {
            x += xb;
            y += yb;
            i += dx;
            j += dy;
         }
      }
   }
   int nx = mask.GetNbinsX();
   int ny = mask.GetNbinsY();
   mask.Despeckle((nx*ny - nborder - 1) / 2);
   if (mask.GetBinContent(1,1) > 0 || mask.GetBinContent(nx,1) > 0 ||
                                      mask.GetBinContent(1,ny) > 0 ||
                                      mask.GetBinContent(nx,ny) > 0)
   {
      Map2D mask2(mask);
      mask2.Flood(2).Mask(&mask,1);
      mask.Add(&mask2);
      mask.Shift(0,0,-1);
   }
   for (int p=0; p<4; ++p) {
      if (refmap == fMap[p]) {
         fMap[p]->Mask(&mask);
         fMap[p]->Draw("colz");
         can->Update();
         if (fZeroMask)
            mask.Mask(fZeroMask);
         setmask(&mask);
      }
   }
}

Map2D *Couples::level(Map2D *map, const char* newname)
{
   // Tilt the surface of map to level it as much as possible.
   // If newname is provided, the original map is left unchanged
   // and a new object containing the leveled map is returned,
   // otherwise the input map is overwritten. Call level multiple
   // times to successively improve the leveling.

   int nxbins = map->GetNbinsX();
   int nybins = map->GetNbinsY();
   double xmax = map->GetXaxis()->GetXmax();
   double xmin = map->GetXaxis()->GetXmin();
   double ymax = map->GetYaxis()->GetXmax();
   double ymin = map->GetYaxis()->GetXmin();
   double x0 = (xmax + xmin)/2;
   double y0 = (ymax + ymin)/2;
   double Suu=0, Sxu=0, Sxx=0;
   double Syu=0, Syy=0, Sxy=0;
   double Sxz=0, Syz=0, Szu=0;
   for (int ix=1; ix <= nxbins; ++ix) {
      double x = map->GetXaxis()->GetBinCenter(ix) - x0;
      for (int iy=1; iy <= nybins; ++iy) {
         double y = map->GetYaxis()->GetBinCenter(iy) - y0;
         double z = map->GetBinContent(ix,iy);
         if (z != 0) {
            Suu += 1;
            Sxu += x;
            Sxx += x*x;
            Syu += y;
            Syy += y*y;
            Sxy += x*y;
            Sxz += x*z;
            Syz += y*z;
            Szu += z;
         }
      }
   }
   double determ = Sxx * (Syy*Suu - Syu*Syu) +
                   Sxy * (Sxu*Syu - Sxy*Suu) +
                   Sxu * (Sxy*Syu - Syy*Sxu);
   double Sinv[3][3];
   Sinv[0][0] = (Syy*Suu - Syu*Syu) / determ;
   Sinv[0][1] = (Syu*Sxu - Sxy*Suu) / determ;
   Sinv[0][2] = (Sxy*Syu - Syy*Sxu) / determ;
   Sinv[1][1] = (Sxx*Suu - Sxu*Sxu) / determ;
   Sinv[1][2] = (Sxy*Sxu - Sxx*Syu) / determ;
   Sinv[2][2] = (Sxx*Syy - Sxy*Sxy) / determ;
   Sinv[1][0] = Sinv[0][1];
   Sinv[2][0] = Sinv[0][2];
   Sinv[2][1] = Sinv[1][2];
   double thetax, thetay, z0;
   thetax = Sinv[0][0]*Sxz + Sinv[0][1]*Syz + Sinv[0][2]*Szu;
   thetay = Sinv[1][0]*Sxz + Sinv[1][1]*Syz + Sinv[1][2]*Szu;
   z0 = Sinv[2][0]*Sxz + Sinv[2][1]*Syz + Sinv[2][2]*Szu;
   if (newname) {
      map = new Map2D(*map);
      map->SetName(newname);
   }
   map->Scale(1e-3);
   map->Rotate(-M_PI/2, -1e-6*thetax, 0);
   map->Rotate(0, -1e-6*thetay, 0);
   map->Scale(1e3);
   return map;
}

Map2D *Couples::zerocurl(const Map2D *gx, const Map2D *gy, Map2D **silhouette)
{
   // Run Map2D::TotalCurl(gx,gy) but then take the result and
   // apply a smooth correction to the resulting contour path
   // integral that makes it sum to zero around the closed loop.
   // The caller becomes owner of the returned object.
 
   Map2D *curlmap = new Map2D(*gx);
   curlmap->TotalCurl(gx, gy, silhouette);

   // TotalCurl generates some 1D histograms as by-products,
   // use them to find a correction that zeros the total curl.

   TH1D *loopcurl = (TH1D*)gROOT->FindObject("loopcurl");
   TH1I *looppixi = (TH1I*)gROOT->FindObject("looppixi");
   TH1I *looppixj = (TH1I*)gROOT->FindObject("looppixj");
   int nsteps = loopcurl->GetNbinsX();
   double ctotal = loopcurl->GetBinContent(nsteps)-1e-6;
   for (int istep=1; istep <= nsteps; ++istep) {
      int i = looppixi->GetBinContent(istep);
      int j = looppixj->GetBinContent(istep);
      double ci = loopcurl->GetBinContent(istep);
      ci -= (ctotal*istep)/nsteps;
      curlmap->SetBinContent(i+1, j+1, ci);
      loopcurl->SetBinContent(istep, ci);
   }
   return curlmap;
}

Map2D *Couples::zerocurl(int px, int py, Map2D **silhouette)
{
   if (fResultsName[px](0,3) != "hmu" or fResultsName[py](0,3) != "hmu") {
      printf("Error in Couples::zerocurl - display a hmu map first!\n");
      return 0;
   }
   return zerocurl(getmap(px), getmap(py), silhouette);
}

Double_t Couples::uncurl(const Map2D *gx, const Map2D *gy, 
                         double chi[2], Map2D **silhouette)
{
   // Rotate the gradient vector (gx,gy) by angle chi, then
   // evaluate Map2D::TotalCurl(gx,gy) and allow the user to
   // adjust chi to zero out the curl.

   double curl=0;
   double dchi=0;
   Map2D *mcurl;
   Map2D *grad[2];
   mcurl = new Map2D(*gx);
   grad[0] = new Map2D(*gx);
   grad[1] = new Map2D(*gx);
   Map2D gxm(*gx), gym(*gy);
   gxm.Mask(&gym);
   gym.Mask(&gxm);
   double chin[2] = {chi[0], chi[1]};
   while (true) {
      chi[0] = chin[0] + dchi * M_PI/180;
      chi[1] = chin[1] + dchi * M_PI/180;
      printf("chi[0], chi[1] = %lf, %lf\n", chi[0], chi[1]);
      grad[0]->Reset();
      grad[1]->Reset();
      grad[0]->Add(&gxm,-cos(chi[0]));
      grad[0]->Add(&gym,-sin(chi[0]));
      grad[1]->Add(&gym,-cos(chi[1]));
      grad[1]->Add(&gxm,+sin(chi[1]));
      curl = mcurl->TotalCurl(grad[0],grad[1],silhouette);
      if (curl == 0) {
         printf("Error in Couples::uncurl - unable to compute curl "
                "on these images, the edges are too fragmented!\n");
         return 0;
      }
      TH1D *loopcurl = (TH1D*)gROOT->FindObject("loopcurl");
      TCanvas *c1 = (TCanvas*)gROOT->FindObject("c1");
      if (c1) {
         c1->cd();
         loopcurl->Draw();
         c1->Update();
      }
      printf("curl=%lf, new chi (degrees, or enter to accept)? ", curl);
      std::cout << std::flush;
      std::string res;
      std::getline(std::cin, res);
      if (res.size() == 0)
         break;
      dchi = std::atof(res.c_str());
   }
   mcurl->SetName("mcurl");
   grad[0]->SetName("gradx");
   grad[1]->SetName("grady");
   mcurl->SetStats(0);
   grad[0]->SetStats(0);
   grad[1]->SetStats(0);
   return curl;
}

Double_t Couples::uncurl(int px, int py, Map2D **silhouette)
{
   if (fResultsName[px](0,3) != "hmu" or fResultsName[py](0,3) != "hmu") {
      printf("Error in Couples::uncurl - display a hmu map first!\n");
      return 0;
   }
   double chi[2] = {fChirot[px], fChirot[py]};
   double curl = uncurl(getmap(px), getmap(py), chi, silhouette);
   fChirot[px] = chi[0];
   fChirot[py] = chi[1];
   return curl;
}

Map2D *Couples::divergence(const Map2D *gx, const Map2D *gy)
{
   // Computes the 2d divergence of the vector field (gx,gy)
   // where the gradient is evaluated in a basis that is
   // rotated by chi with respect to the camera.
 
   Map2D *grad[2];
   grad[0] = new Map2D(*gx);
   grad[1] = new Map2D(*gx);
   Map2D gxm(*gx), gym(*gy);
   gxm.Mask(&gym);
   gym.Mask(&gxm);
   grad[0]->Reset();
   grad[0]->Add(&gxm,-cos(fChirot[0]));
   grad[0]->Add(&gym,-sin(fChirot[0]));
   grad[1]->Reset();
   grad[1]->Add(&gym,-cos(fChirot[1]));
   grad[1]->Add(&gxm,+sin(fChirot[1]));
   grad[0]->SetName("gradx");
   grad[1]->SetName("grady");
   grad[0]->SetStats(0);
   grad[1]->SetStats(0);
   Map2D *mdiv = new Map2D(*gx);
   mdiv->Divergence(&gxm, &gym);
   mdiv->SetName("mdiv");
   mdiv->SetDirectory(0);
   mdiv->SetStats(0);
   return mdiv;
}

Map2D *Couples::divergence(int px, int py)
{
   if (fResultsName[px](0,3) != "hmu" or fResultsName[py](0,3) != "hmu") {
      printf("Error in Couples::divergence - display a hmu map first!\n");
      return 0;
   }
   return divergence(getmap(px), getmap(py));
}

void Couples::drawdiff(const char* name)
{
   // Take the 2d difference between complementary scans

   TString defname("hmu_moco");
   if (name == 0)
      name = defname.Data();
   Map2D *hmu[4];
   for (int p=0; p<4; ++p) {
      hmu[p] = getmap(p, name);
      if (hmu[p] == 0) {
         printf("Error in Couples::drawdiff - no such topograph "
                "named %s, cannot continue.\n", name);
         return;
      }
   }
   for (int d=0; d<2; ++d) {
      hmu[d]->Add(hmu[d+2], -1);
      hmu[d]->Mask(fMap[d]);
      hmu[d]->Mask(fMap[d+2]);
      TString name;
      name.Form("diff%d%d",d,d+2);
      hmu[d]->SetName(name);
      TString title(hmu[d]->GetTitle());
      hmu[d]->SetTitle(title + " difference");
      select_canvas(d);
      hmu[d]->SetContour(100);
      hmu[d]->SetStats(0);
      hmu[d]->Draw("colz");
      double diffmin = hmu[d]->GetMinimum();
      double diffmax = hmu[d]->GetMaximum();
      TH1D *hprof = &hmu[d]->Profile(name + "p", 
                                     title + " difference profile",
                                     1000, diffmin, diffmax);
      select_canvas(d+2);
      gStyle->SetOptFit();
      TFitResultPtr fitres = hprof->Fit("gaus", "qs");
      diffmin = fitres->Parameter(1) - 15*fitres->Parameter(2);
      diffmax = fitres->Parameter(1) + 15*fitres->Parameter(2);
      hmu[d]->SetMinimum(diffmin);
      hmu[d]->SetMaximum(diffmax);
      hprof->GetXaxis()->SetRangeUser(diffmin, diffmax);
      hprof->Draw();
   }
}

Map2D *Couples::surface(const char *name)
{
   // Solve Poisson's equation for the shape of the surface
   // described by the 4 rocking curve topographs with name.

   TString defname("hmu_moco");
   if (name == 0)
      name = defname.Data();
   Map2D *hmu[4];
   for (int p=0; p<4; ++p) {
      hmu[p] = getmap(p, name);
      if (hmu[p] == 0) {
         printf("Error in Couples::surface - getmap returns null "
                "for name=%s\n", name);
         return 0;
      }
   }
   double curl[2];
   Map2D *mcurl[2];
   Map2D *msilh[2]={0,0};
   curl[0] = uncurl(1,0,&msilh[0]);
   mcurl[0] = (Map2D*)gROOT->FindObject("mcurl");
   if (curl[0] == 0 || mcurl[0] == 0) {
      printf("Error in Couples::surface - total curl returns 0!\n");
      return 0;
   }
   mcurl[0]->SetTitle("outer contour 1,0");
   mcurl[0]->SetName("contour10");
   for (int i=0; i<3; i++)
      level(mcurl[0]);
   mcurl[0]->Scale(1e-3);
   msilh[0]->SetTitle("outer contour silhouette 1,0");
   msilh[0]->SetName("silhouette10");
   curl[1] = uncurl(3,2,&msilh[1]);
   mcurl[1] = (Map2D*)gROOT->FindObject("mcurl");
   if (curl[1] == 0 || mcurl[1] == 0) {
      printf("Error in Couples::surface - total curl returns 0!\n");
      return 0;
   }
   mcurl[1]->SetTitle("outer contour 3,2");
   mcurl[1]->SetName("contour32");
   for (int i=0; i<3; i++)
      level(mcurl[1]);
   mcurl[1]->Scale(1e-3);
   msilh[1]->SetTitle("outer contour silhouette 3,2");
   msilh[1]->SetName("silhouette32");
   Map2D *mdive[2];
   mdive[0] = divergence(1,0);
   mdive[0]->SetTitle("divergence 1,0 (1/mm)");
   mdive[0]->SetName("divergence10");
   mdive[0]->SetDirectory(gDirectory);
   mdive[1] = divergence(3,2);
   mdive[1]->SetTitle("divergence 3,2 (1/mm)");
   mdive[1]->SetName("divergence32");
   mdive[1]->SetDirectory(gDirectory);
   Map2D *msurf[2];
   msurf[0] = new Map2D(mcurl[0]);
   msurf[0]->SetName("surface10");
   msurf[0]->SetTitle("(1,0,0) surface 10");
   msurf[0]->SetDirectory(gDirectory);
   msurf[0]->GetXaxis()->SetTitle("x (mm)");
   msurf[0]->GetYaxis()->SetTitle("y (mm)");
   msurf[0]->GetZaxis()->SetTitle("z (#mum)");
   msurf[0]->PoissonSolve(mdive[0]);
   msurf[0]->Mask(msilh[0]);
   msurf[0]->Shift(0,0,-msurf[0]->GetMinimum());
   msurf[0]->Rescale(1,1,1e-3);
   msurf[0]->SetContour(100);
   msurf[1] = new Map2D(mcurl[1]);
   msurf[1]->SetName("surface32");
   msurf[1]->SetTitle("(1,0,0) surface 32");
   msurf[1]->SetDirectory(gDirectory);
   msurf[1]->GetXaxis()->SetTitle("x (mm)");
   msurf[1]->GetYaxis()->SetTitle("y (mm)");
   msurf[1]->GetZaxis()->SetTitle("z (#mum)");
   msurf[1]->PoissonSolve(mdive[1]);
   msurf[1]->Mask(msilh[1]);
   msurf[1]->Shift(0,0,-msurf[1]->GetMinimum());
   msurf[1]->Rescale(1,1,1e-3);
   msurf[1]->SetContour(100);
   return msurf[0];
}

Map2D *Couples::surface(const Map2D *dive, const Map2D *bcon)
{
   // Solve Poisson's equation for the shape of the surface
   // described by the divergence map dive and the boundary
   // conditions map bcon. The dimensions and pixel size of
   // dive and bcon should be the same. The x and y axes of
   // dive and bcon should be in mm, with the zaxis of dive
   // in nm/mm^2, and of bcon in nm. The results are passed
   // back in a new Map2D of the same dimensions and units
   // as bcon. Ownership of the returned object passes to
   // the caller.

   Map2D *msurf = new Map2D(bcon);
   msurf->SetDirectory(gDirectory);
   msurf->PoissonSolve(dive);
   msurf->SetTitle("surface for " + TString(GetTitle()));
   msurf->GetXaxis()->SetTitle("x (mm)");
   msurf->GetYaxis()->SetTitle("y (mm)");
   msurf->GetZaxis()->SetTitle("z (#mum)");
   if (fZeroMask)
      msurf->Mask(getmask());
   msurf->Shift(0,0,-msurf->GetMinimum());
   msurf->Rescale(1,1,1e-3);
   msurf->SetContour(100);
   return msurf;
}

Map2D *Couples::beamspot(int p, double xcenter, double ycenter, 
                                double xsigma, double ysigma)
{
   // Form a 2D gaussian intensity profile centered at (xcenter,ycenter)
   // with rms widths (xsigma,ysigma) such that the 1 sigma contour is
   // an ellipse with major/minor axes aligned with the x/y directions.
   // This map is overlayed on whatever is plotted in graphics pad p,
   // and returned to the user, who is responsible for disposing of it.
   // A new 1D histogram is generated as a by-product under the name
   // beamspot_<p> which contains the weighted sum of the rocking
   // curves for all pixels in scan <p>, weighted by the beamspot
   // intensity, fitted to a single gaussian peak.
   //
   // Note that this method uses the fit result maps hmu,hsigma to
   // generate the integrated rocking curve, in contrast to going back
   // to the original raw single-pixel rocking curves and summing them
   // with weights, as beamspot.C does. Each of these approaches has its
   // own strengths and weaknesses. The beamspot.C approach takes into
   // account the non-gaussian shape of the individual-pixel rocking
   // curves, but it suffers from artificial shaping that comes from
   // intensity variations of the X-ray beam across the sample that
   // distort the weighting by the electron beam intensity. In
   // practice, these differences were observed to be small.

   Map2D *spot;
   while ((spot = (Map2D*)gROOT->FindObject("beamspot")))
      delete spot;
   spot = getmap(p);
   if (spot == 0) {
      printf("Error in Couples::beamspot - use the draw() method to plot "
             "a topograph before trying to superimpose a beamspot on it!\n");
      return 0;
   }
   TString rchname;
   rchname.Form("beamspot_%d",p);
   TH1D *rch;
   while ((rch = (TH1D*)gROOT->FindObject(rchname)))
      delete rch;
   int nsteps = 10000;
   rch = new TH1D(rchname, "beam-weighted whole-crystal rocking curve",
                  nsteps, 0, spot->GetMaximum());
   double step = rch->GetBinWidth(1);
   Couples cotmp(this);
   Map2D *mmu = cotmp.getmap(p,"hmu_moco");
   Map2D *msi = cotmp.getmap(p,"hsigma_moco");
   if (mmu == 0 || msi == 0) {
      printf("Error in Couples::beamspot - unable to find the hmu or hsigma "
             "topographs for *this, cannot continue!\n");
      return 0;
   }
   spot->Reset();
   spot->SetName("beamspot");
   int nxbins = spot->GetNbinsX();
   int nybins = spot->GetNbinsY();
   for (int i=1; i<=nxbins; ++i) {
      double x = spot->GetXaxis()->GetBinCenter(i);
      double x2 = pow((x-xcenter)/xsigma,2);
      for (int j=1; j<=nybins; ++j) {
         double y = spot->GetYaxis()->GetBinCenter(j);
         double y2 = pow((y-ycenter)/ysigma,2);
         double z = exp(-(x2+y2)/2);
         spot->SetBinContent(i,j,z);
         double mu = mmu->Interpolate(x,y);
         double si = msi->Interpolate(x,y);
         if (mu > 0 || si > 0) {
            int k0, k1;
            k0 = (mu - 10*si)/step;
            k1 = (mu + 10*si)/step;
            k0 = (k0 < 1)? 1 : (k0 > nsteps)? nsteps : k0;
            k1 = (k1 < 1)? 1 : (k1 > nsteps)? nsteps : k1;
            for (int k=k0; k<=k1; ++k) {
               double s = (k - 0.5)*step;
               double t = (s - mu)/si;
               rch->Fill(s, z * exp(-t*t/2));
            }
         }
      }
   }
   // superimpose the beamspot image on the topograph 
   TCanvas *can = select_canvas(p);
   TPad *pad1 = new TPad("pad1","", 0,0,1,1);
   TPad *pad2 = new TPad("pad2","", 0,0,1,1);
   pad2->SetFillStyle(4000); // pad2 is transparent
   TExec *ex1 = new TExec("ex1", "Couples::palette1();");
   TExec *ex2 = new TExec("ex2", "Couples::palette2();");
   pad1->Draw();
   pad1->cd();
   ex1->Draw();
   fMap[p]->Draw("colz");
   pad1->Update();
   can->cd();
   double x1,y1,x2,y2;
   pad1->GetRange(x1,y1,x2,y2);
   pad2->Range(x1,y1,x2,y2);
   pad2->Draw();
   pad2->cd();
   ex2->Draw();
   spot->SetMinimum(0.1);
   spot->SetMaximum(1.0);
   spot->Draw("col same");
   pad2->Update();
   ex1->Draw();
   return spot;
}

Map2D *Couples::beamspot2D(int px, int py, double xcenter, double ycenter, 
                                           double xsigma, double ysigma,
                                           double xycorr)
{
   // Form a 2D gaussian beam intensity profile centered at (xcenter,ycenter)
   // with rms widths xsigma along x and ysigma along y, and x-y correlation
   // coefficient xycorr. Taken together, these five parameters describe a
   // general 2D ellipse overlaid on the surface of the diamond. This beam
   // intensity profile is overlayed on the pair px,py of complementary
   // rocking curve topographs and plotted in canvases px and py. It is also
   // used together with the associated hmu,hsigma maps to generate the net
   // probaility density in the plane of tilt angles (theta_x, theta_y). 
   // This angular 2D intensity in the form of a pointer to a new Map2D is
   // is handed back to the caller, who is responsible for disposing of it.
   //
   // Note that this method uses the fit result maps hmu,hsigma to
   // generate the integrated rocking curve, in contrast to going back
   // to the original raw single-pixel rocking curves and summing them
   // with weights, as beamspot.C does. Each of these approaches has its
   // own strengths and weaknesses. The beamspot.C approach takes into
   // account the non-gaussian shape of the individual-pixel rocking
   // curves, but it suffers from artificial shaping that comes from
   // intensity variations of the X-ray beam across the sample that
   // distort the weighting by the electron beam intensity. In
   // practice, these differences were observed to be small.

   Map2D *spot;
   while ((spot = (Map2D*)gROOT->FindObject("beamspot")))
      delete spot;
   Map2D *xspot = getmap(px);
   Map2D *yspot = getmap(py);
   if (xspot == 0 || yspot == 0) {
      printf("Error in Couples::beamspot2D - use the draw() method to plot "
             "the px and py topographs before trying to superimpose a "
             "beamspot on them!\n");
      return 0;
   }
   TString rchname;
   rchname.Form("beamspot_%d,%d", px, py);
   Map2D *rch;
   while ((rch = (Map2D*)gROOT->FindObject(rchname)))
      delete rch;
   int nsteps = 500;
   rch = new Map2D(TH2D(rchname, "beam-weighted whole-crystal rocking curve",
                        nsteps, 0, xspot->GetMaximum(),
                        nsteps, 0, yspot->GetMaximum()));
   double xstep = rch->GetXaxis()->GetBinWidth(1);
   double ystep = rch->GetYaxis()->GetBinWidth(1);
   Couples cotmp(this);
   Map2D *mmux = cotmp.getmap(px, "hmu_moco");
   Map2D *msix = cotmp.getmap(px, "hsigma_moco");
   Map2D *mmuy = cotmp.getmap(py, "hmu_moco");
   Map2D *msiy = cotmp.getmap(py, "hsigma_moco");
   if (mmux == 0 || msix == 0 || mmuy == 0 || msiy == 0) {
      printf("Error in Couples::beamspot2D - unable to find the hmu or hsigma "
             "topographs for *this, cannot continue!\n");
      return 0;
   }
   xspot->Reset();
   int nxbins = xspot->GetNbinsX();
   int nybins = xspot->GetNbinsY();
   if (nxbins != yspot->GetNbinsX() || nybins != yspot->GetNbinsY()) {
      printf("Error in Couples::beamspot2D - expected to find topographs "
             "px and py with the same dimensions, cannot continue!\n");
      return 0;
   }
   double covdeterm = xsigma*xsigma * ysigma*ysigma * (1 - xycorr*xycorr);
   double covarinv[3] = {ysigma*ysigma / covdeterm,
                        -xsigma*ysigma*xycorr / covdeterm,
                         xsigma * xsigma / covdeterm};
   double dcollim = 76e3;
   for (int i=1; i<=nxbins; ++i) {
      double x = xspot->GetXaxis()->GetBinCenter(i);
      for (int j=1; j<=nybins; ++j) {
         double y = xspot->GetYaxis()->GetBinCenter(j);
         double rho2 = pow(x - xcenter, 2) * covarinv[0] +
                       2 * (x - xcenter) * (y - ycenter) * covarinv[1] +
                       pow(y - ycenter, 2) * covarinv[2];
         double z = exp(-0.5 * rho2);
         xspot->SetBinContent(i,j,z);
         double mux = mmux->Interpolate(x,y);
         double six = msix->Interpolate(x,y);
         double muy = mmuy->Interpolate(x,y);
         double siy = msiy->Interpolate(x,y);
         //six = siy = 20; // override with a constant peak width
         // include the effects of beam convergence at the radiator
         int face = -1; // decide which way the diamond is facing, +/-1
         mux += face * (x - xcenter) / dcollim;
         muy += face * (y - ycenter) / dcollim;
         if (mux > 0 || six > 0) {
            int k0, k1;
            k0 = (mux - 10*six)/xstep;
            k1 = (mux + 10*six)/xstep;
            k0 = (k0 < 1)? 1 : (k0 > nsteps)? nsteps : k0;
            k1 = (k1 < 1)? 1 : (k1 > nsteps)? nsteps : k1;
            for (int k=k0; k<=k1; ++k) {
               double sx = (k - 0.5)*xstep;
               double tx = (sx - mux)/six;
               if (muy > 0 || siy > 0) {
                  int l0, l1;
                  l0 = (muy - 10*siy)/ystep;
                  l1 = (muy + 10*siy)/ystep;
                  l0 = (l0 < 1)? 1 : (l0 > nsteps)? nsteps : l0;
                  l1 = (l1 < 1)? 1 : (l1 > nsteps)? nsteps : l1;
                  for (int l=l0; l<=l1; ++l) {
                     double sy = (l - 0.5)*ystep;
                     double ty = (sy - muy)/siy;
                     rch->Fill(sx, sy, z * exp(-0.5*(tx*tx + ty*ty)));
                  }
               }
            }
         }
      }
   }
   // superimpose the beamspot image on the topographs
   double x1,y1,x2,y2;
   TCanvas *canx = select_canvas(px);
   TPad *padx1 = new TPad("padx1","", 0,0,1,1);
   TPad *padx2 = new TPad("padx2","", 0,0,1,1);
   padx2->SetFillStyle(4000); // padx2 is transparent
   TExec *ex1 = new TExec("ex1", "Couples::palette1();");
   TExec *ex2 = new TExec("ex2", "Couples::palette2();");
   padx1->Draw();
   padx1->cd();
   ex1->Draw();
   fMap[px]->Draw("colz");
   padx1->Update();
   canx->cd();
   padx1->GetRange(x1,y1,x2,y2);
   padx2->Range(x1,y1,x2,y2);
   padx2->Draw();
   padx2->cd();
   ex2->Draw();
   xspot->SetMinimum(0.1);
   xspot->SetMaximum(1.0);
   xspot->Draw("col same");
   padx2->Update();
   ex1->Draw();

   TCanvas *cany = select_canvas(py);
   TPad *pady1 = new TPad("pady1","", 0,0,1,1);
   TPad *pady2 = new TPad("pady2","", 0,0,1,1);
   pady2->SetFillStyle(4000); // pady2 is transparent
   TExec *ey1 = new TExec("ey1", "Couples::palette1();");
   TExec *ey2 = new TExec("ey2", "Couples::palette2();");
   pady1->Draw();
   pady1->cd();
   ey1->Draw();
   fMap[py]->Draw("colz");
   pady1->Update();
   cany->cd();
   pady1->GetRange(x1,y1,x2,y2);
   pady2->Range(x1,y1,x2,y2);
   pady2->Draw();
   pady2->cd();
   ey2->Draw();
   xspot->SetMinimum(0.1);
   xspot->SetMaximum(1.0);
   xspot->Draw("col same");
   pady2->Update();
   ey1->Draw();

   canx->Update();
   cany->Update();
   canx->Update();
   return xspot;
}

void Couples::palette1()
{
   select_palette(0);
}

void Couples::palette2()
{
   gStyle->SetPalette(kInvertedDarkBodyRadiator,0,0.6);
}

void Couples::remove_walk(Map2D *target, Map2D *mmu, double dydtheta)
{
   // Private method to remove target walk along y with theta
   // that is the result of the target not being centered on
   // the rotation axis of the theta stage. The target map
   // is overwritten with the result. This is essentially
   // just the Fill method of Map2D with a slight twist.

   Map2D source(target);
   target->Reset();
   int nxbins = target->GetXaxis()->GetNbins();
   int nybins = target->GetYaxis()->GetNbins();
   double ymin = target->GetYaxis()->GetXmin();
   double ymax = target->GetYaxis()->GetXmax();
   double mu0 = (mmu->GetMaximum() + mmu->GetMinimum())/2;
   for (Int_t i=1; i <= nxbins; ++i) {
      double x = target->GetXaxis()->GetBinCenter(i);
      double yprime[nybins+2];
      yprime[0] = 0;
      for (Int_t j=1; j <= nybins; ++j) {
         double y = target->GetYaxis()->GetBinCenter(j);
         double mu = mmu->Interpolate(x,y);
         yprime[j] = y + dydtheta*(mu - mu0);
      }
      yprime[nybins+1] = DBL_MAX;
      int jprime = 0;
      for (Int_t j=1; j <= nybins; ++j) {
         double y = target->GetYaxis()->GetBinCenter(j);
         while (y > yprime[jprime+1])
            ++jprime;
         double f1 = y - yprime[jprime];
         double f0 = yprime[jprime+1] - y;
         double y0 = target->GetYaxis()->GetBinCenter(jprime);
         double y1 = target->GetYaxis()->GetBinCenter(jprime+1);
         double yy = (f0*y0 + f1*y1)/(f0 + f1);
         if (yy > ymin && yy < ymax) {
            double z = source.Interpolate(x,yy);
            target->SetBinContent(i,j,z);
         }
      }
   }
}

Map2D *Couples::getmmu(int p)
{
   // Returns the mu map used for dy/dtheta walk correction,
   // generating a new one if it does not already exist for
   // scan p. The mu map is a smoothed version of the hmu
   // peak mean topograph for the given scan that has been
   // smoothed and had its voids filled in to make it give
   // smooth results for the dy/dtheta correction. Look at
   // method remove_walk for details on that algorithm.

   if (fMmu[p] == 0) {
      TFile *fresults = open(fResultsPath[p]);
      if (fresults == 0)
         return 0;
      TH2D *hmu = (TH2D*)fresults->Get("hmu");
      if (hmu == 0)
         return 0;
      Map2D *mmu = new Map2D(*hmu);
      if (fCleanMask[p])
         mmu->Mask(fCleanMask[p]);
      mmu->Despike(100);
      mmu->FillVoids();
      mmu->Smooth();
      TString name;
      mmu->SetName(name.Format("mmu%d", p));
      fMmu[p] = mmu;
   }
   return fMmu[p];
}

void Couples::setmmu(int p, const Map2D *mmu)
{
   if (fMmu[p]) {
      delete fMmu[p];
      fMmu[p] = 0;
   }
   if (mmu)
      fMmu[p] = new Map2D(mmu);
}

Double_t Couples::correlation(int p0, int p1) const
{
   // Find the correlation between the current topographs
   // from scans p0 and p1.

   if (fMap[p0] == 0 || fMap[p1] == 0) {
      printf("Error in Couples::correlation - please call draw "
             "to select the topographs before computing their "
             "correlation.\n");
      return 0;
   }
   return fMap[p0]->Correlation(fMap[p1]);
}

void Couples::crosscheck(const Map2D *surf, const Map2D *gx, const Map2D *gy)
{
   // Takes a geometric solution for the (1,0,0) surface shape surf
   // and computes the x and y gradients, then compares them to the
   // input maps gx,gy. The comparision is made first in terms of
   // their correlation, then in terms of the rms difference. It is
   // assumed that the x,y dimensions and resolutions of all three
   // input maps is the same, and that the z axis of surf is in
   // units of microns, while gx,gy are in units of urad.
 
   Map2D *sgx = new Map2D(gx);
   Map2D *sgy = new Map2D(gy);
   sgx->SetName("sgx");
   sgy->SetName("sgy");
   sgx->Reset();
   sgy->Reset();
   surf->Gradient(sgx,sgy);
   Map2D gxmasked(gx);
   Map2D gymasked(gy);
   gxmasked.Mask(sgx);
   gymasked.Mask(sgy);
   printf("gradients give correlation coefficients x,y=%lf,%lf\n",
          sgx->Correlation(&gxmasked), sgy->Correlation(&gymasked));
   sgx->Add(&gxmasked,-1e-3);
   select_canvas(0);
   sgx->SetContour(100);
   sgx->Draw("colz");
   sgy->Add(&gymasked,-1e-3);
   select_canvas(2);
   sgy->SetContour(100);
   sgy->Draw("colz");
   TH1D &dsgx = sgx->Profile("dsgx","",10000,-10,10);
   TH1D &dsgy = sgy->Profile("dsgy","",10000,-10,10);
   select_canvas(1);
   TFitResultPtr dsgxfit = dsgx.Fit("gaus","sq");
   select_canvas(3);
   TFitResultPtr dsgyfit = dsgy.Fit("gaus","sq");
   printf("rms differences are x,y=%lf,%lf\n",dsgx.GetRMS(),dsgy.GetRMS());
   printf("difference sigmas are x,y=%lf,%lf\n",
          dsgxfit->Parameter(2), dsgyfit->Parameter(2));
}

TFile *Couples::open(const char *filename)
{
   if (strlen(filename) == 0)
      return 0;
   int n;
   TString path;
   for (n=0; n<nsearchpaths; ++n) {
      path = TString(searchpath[n]) + TString(filename);
      ifstream fin(path.Data());
      if (fin.good())
         break;
   }
   if (n < nsearchpaths) {
      TDirectory *cwd(gDirectory);
      TFile *file = TFile::Open(path.Data());
      cwd->cd();
      return file;
   }
   printf("Error in Couples::open - unable to open input file %s\n", filename);
   return 0;
}

void Couples::dump()
{
   printf("Contents of Couples %s: %s\n", GetName(), GetTitle());
   printf("   images are %d x %d pixels\n", fPixels[1], fPixels[0]);
   printf("              [%lf, %lf] mm horizontal\n", fXrange[0], fXrange[1]);
   printf("              [%lf, %lf] mm vertical\n", fYrange[0], fYrange[1]);
   if (fZeroMask)
      printf("   mask is enabled (%s)\n", fZeroMask->GetName());
   else
      printf("   mask is disabled\n");
   for (int p=0; p<4; ++p) {
      if (fResultsPath[p].Length() == 0)
         printf("   couple member %d: (none)\n", p+1);
      else {
         printf("   couple member %d: %s ", p+1, fResultsPath[p].Data());
         if (fResultsName[p].Length() == 0)
            printf("\n");
         else
            printf("(%s)\n", fResultsName[p].Data());
         printf("      cleaning mask ");
         if (fCleanMask[p])
            printf("%s (%s)\n", fCleanMask[p]->GetName(),
                                fCleanMask[p]->GetTitle());
         else
            printf("(none)\n");
         printf("      shifted (%lf, %lf) mm\n", fXshift[p], fYshift[p]);
         printf("      rotated %lf degrees\n", fPsirot[p] * 180/M_PI);
         printf("      rescaled by (%lf, %lf)\n", fXscale[p], fYscale[p]);
         printf("      x/y skew factor %lf\n", fXYskew[p]);
         printf("      chi angle %lf degrees\n", fChirot[p] * 180/M_PI);
         printf("      phi angle %lf degrees\n", fPhirot[p] * 180/M_PI);
         printf("      residual x,y dispersion %lf,%lf urad/mm\n",
                       fXdisper[p], fYdisper[p]);
         printf("      dy/dtheta image walk %lf mm/mrad\n", fYwalk[p]);
         printf("      current image: ");
         if (fMap[p])
            printf("%s (%s)\n", fMap[p]->GetName(), fMap[p]->GetTitle());
         else
            printf("(none)\n");
      }
   }
}

void Couples::display(Map2D *m1, Map2D *m2, Map2D *m3, Map2D *m4)
{
   // Convenience method for plotting up to 4 user maps 
   // in the 4 graphics windows managed by this class.

   Map2D *map[4] = {m1,m2,m3,m4};
   for (int p=0; p<4; ++p) {
      select_canvas(p);
      if (map[p]->InheritsFrom("Map2D")) {
         map[p]->SetContour(100);
         map[p]->Draw("colz");
      }
      else {
         map[p]->Draw();
      }
   }
}

void Couples::draw(const char *name)
{
   // Draw the two images of the Couples with the given name, or the
   // last requested images if name is not given.

   for (int p=0; p<4; ++p) {
      TCanvas *can = select_canvas(p);
      if (getmap(p, name)) {
         fMap[p]->SetContour(100);
         fMap[p]->Draw("colz");
      }
   }
}

TCanvas *Couples::select_canvas(int p)
{
   // Select one of four square drawing areas for subsequent drawing.

   gStyle->SetCanvasPreferGL(true);
   TCanvas *can;
   if (p == 0) {
      can = (TCanvas*)gROOT->FindObject("c1");
      if (can == 0) {
         can = new TCanvas("c1", "c1", 0, 0, 560, 500);
         can->SetRightMargin(0.15);
      }
   }
   else if (p == 1) {
      can = (TCanvas*)gROOT->FindObject("c2");
      if (can == 0) {
         can = new TCanvas("c2", "c2", 0, 600, 560, 500);
         can->SetRightMargin(0.15);
      }
   }
   else if (p == 2) {
      can = (TCanvas*)gROOT->FindObject("c3");
      if (can == 0) {
         can = new TCanvas("c3", "c3", 562, 0, 560, 500);
         can->SetRightMargin(0.15);
      }
   }
   else {
      can = (TCanvas*)gROOT->FindObject("c4");
      if (can == 0) {
         can = new TCanvas("c4", "c4", 562, 600, 560, 500);
         can->SetRightMargin(0.15);
      }
   }
   can->cd();
   return can;
}

void Couples::save(const char *name, TCanvas *canvas, const char *plotfile)
{
   TH1 *h = (TH1*)gDirectory->Get(name);
   if (h == 0) {
      printf("Error in save - object named %s not found!\n", name);
      return;
   }
   h->Write();
   if (canvas) {
      canvas->cd();
      if (h->InheritsFrom("Map2D")) {
         Map2D *map = (Map2D*)h;
         map->SetContour(100);
         map->Draw("colz");
      }
      else
         h->Draw("colz");
      canvas->Update();
      if (plotfile)
         canvas->Print(plotfile);
   }
}

void Couples::fitandsave(const char *name)
{
   TString defname("hmu_moco");
   if (name == 0)
      name = defname.Data();
   draw(name);
   if (surface(name) == 0) {
      printf("Error in Couples::fitandsave - "
             "Couples::surface returns with error, "
             "cannot continue\n");
      return;
   }
   Map2D *surface10 = (Map2D*)gROOT->FindObject("surface10");
   Map2D *surface32 = (Map2D*)gROOT->FindObject("surface32");
   if (surface10 == 0 || surface32 == 0) {
      printf("Error in Couples::fitandsave - "
             "Couples::surface returns ok, but no surface10 "
             "or surface32 results are found, cannot continue\n");
      return;
   }
   for (int i=0; i<3; i++) {
      level(surface10);
      level(surface32);
   }
   save("surface10", select_canvas(0), "surface10.png");
   save("surface32", select_canvas(2), "surface32.png");
   Map2D *sdiff = new Map2D(surface10);
   sdiff->SetName("surfacediff");
   TString title(sdiff->GetTitle());
   sdiff->SetTitle(title + " difference");
   sdiff->Add(surface32,-1);
   sdiff->Mask((Map2D*)gROOT->FindObject("silhouette10"));
   sdiff->Mask((Map2D*)gROOT->FindObject("silhouette32"));
   save("surfacediff", select_canvas(1), "surfacediff.png");
   TH1D &sdiffpro = sdiff->Profile("surfacediffpro", 
                                   title + " difference profile",
                                   500, sdiff->GetMinimum(),
                                   sdiff->GetMaximum());
   TFitResultPtr fres = sdiffpro.Fit("gaus", "s");
   if (fres->Status()) {
      double mu = fres->Parameter(1);
      double sig = fres->Parameter(2);
      sdiffpro.GetXaxis()->SetRangeUser(mu-10*sig, mu+10*sig);
   }
   save("surfacediffpro", select_canvas(3), "surfacediffpro.png");
   save("contour10");
   save("silhouette10");
   save("divergence10");
   save("contour32");
   save("silhouette32");
   save("divergence32");
   drawdiff();
   save("diff02", select_canvas(0), "diff02.png");
   save("diff13", select_canvas(1), "diff13.png");
   save("diff02p", select_canvas(2), "diff02p.png");
   save("diff13p", select_canvas(3), "diff13p.png");
}
