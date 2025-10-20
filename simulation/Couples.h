//
// Couples.C - header for Couples class.
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
//        * association of complementary pairs                          //
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

#ifndef COUPLE_H
#define COUPLE_H

#include <TFile.h>
#include <TNamed.h>
#include <TCanvas.h>
#include "Map2D.h"

class Couples: public TNamed
{
 public:
   Couples();
   Couples(const char *name, const char *title);
   Couples(const Couples &src);
   Couples(const Couples *src) : Couples(*src) {}
   virtual ~Couples();

   Couples &operator=(const Couples &src);

   void select(int p, const char *resultsfile);
   void setresol(int xpixels, int ypixels=0);
   void setxrange(double xmax, double xmin=0);
   void setyrange(double ymax, double ymin=0);
   void shift(int p, double xshift, double yshift);
   void rotate(int p, double phi, int deg=0);
   void invert(int p, int ix, int iy);
   void zoom(int p, double fzoom);
   void setmin(int p=-1, double min=0);
   void setmax(int p, double max);
   void stretch(int p, double aspectratio);
   void setclean(int p, Map2D *mask);
   void setmask(const Map2D *mask);
   void lineup(int p0, int p1);
   void fillvoids(int p=-1);

   Map2D *getmask();
   Map2D *getclean(int p);
   Map2D *getmap(int p, const char *name=0);
   Double_t getchi(int p) const;
   void setchi(int p, double chi, int deg=0);
   Double_t getphi(int p) const;
   void setphi(int p, double phi, int deg=0);
   Int_t getdispersion(int p, double &xslope, double &yslope) const;
   void setdispersion(int p, double xslope, double yslope=0);
   Double_t getywalk(int p) const;
   void setywalk(int p, double dydtheta);
   Map2D *getmmu(int p);
   void setmmu(int p, const Map2D *mmu);
   Map2D *reshape(const Map2D *src) const;

   Map2D *cleanup(int p, int minspeck=0);
   void polycrop(int p, const char *name=0);
   Map2D *level(Map2D *map, const char* newname=0);
 
   Double_t uncurl(int px, int py, Map2D **silhouette=0);
   Double_t uncurl(const Map2D *gx, const Map2D *gy, 
                   double chi[2], Map2D **silhouette=0);
   Map2D *zerocurl(int px, int py, Map2D **silhouette=0);
   Map2D *zerocurl(const Map2D *gx, const Map2D *gy, Map2D **silhouette=0);
   Map2D *divergence(int px, int py);
   Map2D *divergence(const Map2D *gx, const Map2D *gy);
   Map2D *surface(const char *name=0);
   Map2D *surface(const Map2D *dive, const Map2D *bcon);
   Map2D *beamspot(int p, double xcenter, double ycenter, 
                          double xsigma=1.0, double ysigma=0.7); 
   Map2D *beamspot2D(int px, int py, double xcenter, double ycenter, 
                                     double xsigma=1.0, double ysigma=0.7,
                                     double xycorr=0); 
   Double_t correlation(int p0, int p1) const;
   void crosscheck(const Map2D *surf, const Map2D *gx, const Map2D *gy);

   void display(Map2D *m1, Map2D *m2=0, Map2D *m3=0, Map2D *m4=0);
   void drawdiff(const char *name=0);
   void draw(const char *name=0);
   void dump();

   static void select_palette(int pal);

   ClassDef(Couples, 3);

 protected:
   TString fResultsPath[4];           // path to XXX_results.root
   TString fResultsName[4];           // name of Map2D in raw image set
   Double_t fXshift[4];               // offset x (mm)
   Double_t fYshift[4];               // offset y (mm)
   Double_t fPsirot[4];               // rotation psi (rad)
   Double_t fXscale[4];               // x scale factor
   Double_t fYscale[4];               // y scale factor
   Double_t fXYskew[4];               // skew factor (x/y)
   Double_t fChirot[4];               // chi angle of scan (rad)
   Double_t fPhirot[4];               // phi angle of scan (rad)
   Double_t fXdisper[4];              // x dispersion factor (urad/mm)
   Double_t fYdisper[4];              // y dispersion factor (urad/mm)
   Map2D *fCleanMask[4];              // clean-up masks for raw images
   Map2D *fZeroMask;                  // zero mask for all images
   Map2D *fMap[4];                    // current maps 

   Double_t fXrange[2];               // x axis range low, high (mm)
   Double_t fYrange[2];               // y axis range low, high (mm)
   Int_t fPixels[2];                  // height, width (pixels)

   Double_t fYwalk[4];                // dy/dtheta walk factor (mm/mrad)
   Map2D *fMmu[4];                    // mu map for walk correction

   std::vector<std::pair<int,int> > fPolyVertex; //! cropping polygon

 private:
   void copy(const Couples &src);
   TFile *open(const char *filename);
   TCanvas *select_canvas(int p);
   void polycropper(int p);
   void deletemap(int p) {
      if (fMap[p]) {
         delete fMap[p];
         fMap[p] = 0;
      }
   }

 public:
   static void palette1();
   static void palette2();
   static void polypicker(int p);
   static Couples *fPicking;
   void remove_walk(Map2D *target, Map2D *hmu, double dydtheta);
   void save(const char *name, TCanvas *canvas=0, const char *plotfile=0);
   void fitandsave(const char *name=0);
};

#endif
