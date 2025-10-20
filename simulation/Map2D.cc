/*************************************************************************
 * Copyright(c) 1998, University of Connecticut, All rights reserved.    *
 * Author: Richard T. Jones, Assoc. Prof. of Physics                     *
 *                                                                       *
 * Permission to use, copy, modify and distribute this software and its  *
 * documentation for non-commercial purposes is hereby granted without   *
 * fee, provided that the above copyright notice appears in all copies   *
 * and that both the copyright notice and this permission notice appear  *
 * in the supporting documentation. The author makes no claims about the *
 * suitability of this software for any purpose.                         *
 * It is provided "as is" without express or implied warranty.           *
 *************************************************************************/
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Map2D Class                                                          //
//                                                                      //
// The Map2D class provides a few useful methods for manipulating and   //
// comparing topographical maps within the root framework.  Map2D is    //
// derived from the basic two dimensional histogram TH2D class in root  //
// and inherits all of the storage and visualization capabilities of    //
// the root framework.  The main thing that is missing in TH2D is any   //
// concept of empty or invalid regions within the plot.  Many standard  //
// image formats incorporate "matte" information (or alpha channel) to  //
// indicate regions within the image frame that are not actually part   //
// of the image.  This is a basic feature of maps, that the coverage    //
// region is complex, and cannot be reduced to a rectangle.  The Map2D  //
// class introduces an alpha channel by defining the value (double)0.0  //
// as the matte value, or undefined.  Operations on that value should   //
// leave it undefined, for example: 0 + 3.8 = 0.  Interpolation on a    //
// map in a region next to a matte pixel must be designed so as not to  //
// treat it as z=0, and preserve the sharpness of the boundary between  //
// matte and ordinary pixels.                                           //
//                                                                      //
// The class currently supports the following operations on maps:       //
//        * addition/subtraction of 2 maps,                             //
//        * shifting a map in all 3 dimensions,                         //
//        * rescaling a map in all 3 dimensions,                        //
//        * rotating a map in all 3 dimensions,                         //
//        * computing correlations between 2 maps,                      //
//        * determining the optimum alignment between 2 similar maps,   //
// all of which respect the meaning of matte pixels.                    //
//                                                                      //
// This package was developed at the University of Connecticut by       //
// Richard T. Jones                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#define REPORT_PROCESS_TIME 1

#include <iostream>
#include <TROOT.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TRandom3.h>
#include "Map2D.h"
#include <list>
#include <limits>

#include <TMatrixD.h>
#include <TDecompSVD.h>
#include <TFFTComplexReal.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <map>

ClassImp(Map2D);

#ifndef SQR_FUNC
#define SQR_FUNC 1
Double_t sqr(Double_t x) { return x*x; }
#endif

const Double_t Map2D::zunit2xy = 0.001;

TRandom3 *randoms = 0;

#define EMBED_BREAKPOINT  asm volatile ("int3;")

Map2D::Map2D() : TH2D()
{
   SetStats(0);
}

Map2D::Map2D(const Map2D &m) : TH2D((TH2D&)m)
{
   SetStats(0);
}

Map2D::Map2D(const TH2D &h) : TH2D(h)
{
   SetStats(0);
}

Map2D::~Map2D()
{ }

Map2D &Map2D::operator=(const Map2D &src)
{
   int id = GetUniqueID();
   (TH2D&)*this = (TH2D&)src;
   SetUniqueID(id);
   return *this;
}

Map2D &Map2D::Add(const Map2D *map, Double_t c)
{
   // Overloads TH2D::Add, has the same basic behaviour,
   // except that zero plus anything is zero.

   Int_t nxbins = map->GetXaxis()->GetNbins();
   Int_t nybins = map->GetYaxis()->GetNbins();
   if (nxbins != GetXaxis()->GetNbins() ||
       nybins != GetYaxis()->GetNbins()) {
      std::cerr << "Error in Map2D::Add - source and destination histograms"
                << " have different dimensions, cannot add them"
                << std::endl;
      return *this;
   }
   for (Int_t i=1; i <= nxbins; ++i) {
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t z1 = map->GetBinContent(i,j);
         Double_t z0 = GetBinContent(i,j);
         SetBinContent(i,j,z0+c*z1);
      }
   }
   return *this;
}

Map2D &Map2D::Add(const Map2D *map1, const Map2D *map2,
                          Double_t c1, Double_t c2)
{
   // Overloads TH2D::Add, has the same basic behaviour,
   // except that zero plus anything is zero.

   Int_t nxbins = map1->GetXaxis()->GetNbins();
   Int_t nybins = map1->GetYaxis()->GetNbins();
   if (nxbins != map2->GetXaxis()->GetNbins() ||
       nybins != map2->GetYaxis()->GetNbins()) {
      std::cerr << "Error in Map2D::Add - two source histograms"
                << " have different dimensions, cannot add them"
                << std::endl;
      return *this;
   }
   else if (nxbins != GetXaxis()->GetNbins() ||
            nybins != GetYaxis()->GetNbins()) {
      Double_t xmin = map1->GetXaxis()->GetXmin();
      Double_t xmax = map1->GetXaxis()->GetXmax();
      Double_t ymin = map1->GetYaxis()->GetXmin();
      Double_t ymax = map1->GetYaxis()->GetXmax();
      SetBins(nxbins,xmin,xmax,nybins,ymin,ymax);
   }
   for (Int_t i=1; i <= nxbins; ++i) {
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t z1 = map1->GetBinContent(i,j);
         Double_t z2 = map2->GetBinContent(i,j);
         SetBinContent(i,j,c1*z1+c2*z2);
      }
   }
   return *this;
}

Map2D &Map2D::Shift(Double_t dx, Double_t dy, Double_t dz)
{
   // Translates the map image by a distance (dx,dy) in map coordinates,
   // and shifts all pixels up or down by vertical distance dz in
   // height units.  Image regions shifted outside the range of the
   // map are lost.  Image regions shift into the range of the map
   // from outside are zero.  Original contents are overwritten.

   Map2D *tmp = shift(dx,dy,dz);
   *this = *tmp;
   delete tmp;
   return *this;
}

Map2D *Map2D::shift(Double_t dx, Double_t dy, Double_t dz) const
{
   // Same as Shift(), except a new object is created to hold
   // the results and returned, and the original object is not
   // touched.  The caller must delete the returned object.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dbinx = dx/(xmax-xmin)*nxbins;
   Double_t dbiny = dy/(ymax-ymin)*nybins;
   Int_t di = ceil(dbinx);
   Int_t dj = ceil(dbiny);
   Int_t i0 = (di > 0)? di : 1;
   Int_t j0 = (dj > 0)? dj : 1;
   Int_t i1 = (di < 0)? di+nxbins+1 : nxbins;
   Int_t j1 = (dj < 0)? dj+nybins+1 : nybins;
   Double_t fx1 = di-dbinx;
   Double_t fy1 = dj-dbiny;
   Double_t fx0 = 1-fx1;
   Double_t fy0 = 1-fy1;

   Map2D *dst = new Map2D(*this);
   dst->Reset();
   for (Int_t i=i0; i <= i1; ++i) {
      Int_t isrc = i-di;
      Double_t z00 = GetBinContent(isrc,j0-dj);
      Double_t z10 = GetBinContent(isrc+1,j0-dj);
      Double_t zint0 = (z00 == 0)? ((fx0 > 0.5)? 0 : z10) :
                       (z10 == 0)? ((fx1 > 0.5)? 0 : z00) :
                       z00*fx0 + z10*fx1;
      for (Int_t j=j0; j <= j1; ++j) {
         Int_t jsrc = j-dj;
         Double_t z01 = GetBinContent(isrc,jsrc+1);
         Double_t z11 = GetBinContent(isrc+1,jsrc+1);
         Double_t zint1 = (z01 == 0)? ((fx0 > 0.5)? 0 : z11) :
                          (z11 == 0)? ((fx1 > 0.5)? 0 : z01) :
                          z01*fx0 + z11*fx1;
         Double_t zint = (zint0 == 0)? ((fy0 > 0.5)? 0 : zint1) :
                         (zint1 == 0)? ((fy1 > 0.5)? 0 : zint0) :
                         zint0*fy0 + zint1*fy1;
         dst->SetBinContent(i,j,(zint == 0)? 0 : zint+dz);
         zint0 = zint1;
      }
   }
   return dst;
}

Map2D &Map2D::Rotate(Double_t phi, Double_t theta, Double_t psi, Int_t degrees)
{
   // Rotates the map image by Euler angles phi,theta,psi about the center
   // of the map surface.  Angles may be in degrees (degrees=1) or
   // radians (degrees=0).  The angle phi determines the direction of
   // the theta rotation axis in the xy plane, and does not generate
   // an independent rotation.  Map rotations only make sense for small
   // theta.  The theta rotation is implemented as a rotation about the
   // x axis by angle theta*cos(phi) followed by a rotation about the y
   // axis by angle theta*sin(phi), with errors entering at the level of
   // theta^3.  Rotations in the xy plane (angle psi) may be of any size,
   // and may result in clipping of the image.  Regions of the map that
   // are rotated outside the map range are lost.  Original contents of
   // the map are overwritten.

   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t x0 = (xmin+xmax)/2;
   Double_t y0 = (ymin+ymax)/2;
   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Int_t pcount=0;
   for (Int_t ix=1; ix <= nxbins; ++ix) {
      for (Int_t iy=1; iy <= nybins; ++iy) {
         if (GetBinContent(ix,iy)) {
            ++pcount;
         }
      }
   }
   Int_t i0 = nxbins/2;
   Int_t j0 = nybins/2;
   Int_t peri=0;
   Double_t sum0=1;
   Double_t sum1=GetBinContent(i0,j0);
   while (sum0 < pcount/10) {
      ++peri;
      for (Int_t i=-peri; i < peri; ++i) {
          Double_t z;
          sum1 += ((z=GetBinContent(i0+i,j0-peri)) != 0)? ++sum0,z : 0;
          sum1 += ((z=GetBinContent(i0+peri,j0+i)) != 0)? ++sum0,z : 0;
          sum1 += ((z=GetBinContent(i0-i,j0+peri)) != 0)? ++sum0,z : 0;
          sum1 += ((z=GetBinContent(i0-peri,j0-i)) != 0)? ++sum0,z : 0;
      }
   }
   Double_t z0 = sum1/sum0;
   Double_t toradians = (degrees)? M_PI/180 : 1;
   Double_t thetax = theta*cos(phi*toradians);
   Double_t thetay = theta*sin(phi*toradians);
   Map2D *tmp1 = this;
   if (thetax != 0) {
      tmp1 = rotateX(thetax*toradians,y0,z0);
   }
   Map2D *tmp2 = tmp1;
   if (thetay != 0) {
      tmp2 = tmp1->rotateY(thetay*toradians,x0,z0);
   }
   Map2D *tmp3 = tmp2;
   if (psi != 0) {
      tmp3 = tmp2->rotateZ(psi*toradians,x0,y0);
   }
   if (tmp3 != this) {
      *this = *tmp3;
   }
   if (tmp1 != this) {
      delete tmp1;
   }
   if (tmp2 != tmp1) {
      delete tmp2;
   }
   if (tmp3 != tmp2) {
      delete tmp3;
   }
   return *this;
}

Map2D *Map2D::rotateX(Double_t thetarad, Double_t y0, Double_t z0) const
{
   // Rotates the map image about the x axis by angle thetarad in radians.
   // The rotation axis is located at coordinate y0 and height z0.
   // The map must remain single-valued after the rotation, which places
   // an upper limit on the magnitude of theta that depends on the
   // height profile of the map relative to z0.  An error message is
   // printed if this constraint is violated.  A new Map2D object is
   // created to hold the result, and the original object is untouched.
   // The caller must delete the returned object.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dy = (ymax-ymin)/nybins;
   Double_t costheta = cos(thetarad);
   Double_t sintheta = sin(thetarad);
   Map2D *dst = new Map2D(*this);
   dst->Reset();
   int warnings_printed=0;
   for (Int_t j=1; j <= nybins; ++j) {
      Double_t y = ymin+(j-0.5)*(ymax-ymin)/nybins - y0;
      for (Int_t i=1; i <= nxbins; ++i) {
         Double_t z = GetBinContent(i,j);
         if (z == 0) {
            dst->SetBinContent(i,j,0);
         }
         else {
            dst->SetBinContent(i,j,y*sintheta/zunit2xy+(z-z0)*costheta+z0);
         }
         if (fabs((z-z0)*sintheta)*zunit2xy > dy) {
            if (warnings_printed++ == 0) {
               std::cerr << "Warning in Map2D::rotateX - loss of accuracy"
                         << " trying to rotate too far away from normal."
                         << std::endl;
            }
         }
      }
   }
   Map2D *redst = dst->rescale(1,costheta,1,0,y0,0);
   delete dst;
   return redst;
}

Map2D *Map2D::rotateY(Double_t thetarad, Double_t x0, Double_t z0) const
{
   // Rotates the map image about the y axis by angle thetarad in radians.
   // The rotation axis is located at coordinate x0 and height z0.
   // The map must remain single-valued after the rotation, which places
   // an upper limit on the magnitude of theta that depends on the
   // height profile of the map relative to z0.  An error message is
   // printed if this constraint is violated.  A new Map2D object is
   // created to hold the result, and the original object is untouched.
   // The caller must delete the returned object.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t costheta = cos(thetarad);
   Double_t sintheta = sin(thetarad);
   Map2D *dst = new Map2D(*this);
   dst->Reset();
   for (Int_t i=1; i <= nxbins; ++i) {
      Double_t x = xmin+(i-0.5)*(xmax-xmin)/nxbins - x0;
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t z = GetBinContent(i,j);
         if (z == 0) {
            dst->SetBinContent(i,j,0);
         }
         else {
            dst->SetBinContent(i,j,-x*sintheta/zunit2xy+(z-z0)*costheta+z0);
         }
         if (fabs((z-z0)*sintheta)*zunit2xy > dx) {
            std::cerr << "Warning in Map2D::rotateY - loss of accuracy"
                      << " trying to rotate too far away from normal."
                      << std::endl;
         }
      }
   }
   Map2D *redst = dst->rescale(costheta,1,1,x0,0,0);
   delete dst;
   return redst;
}

Map2D *Map2D::rotateZ(Double_t phirad, Double_t x0, Double_t y0) const
{
   // Rotates the map image about the z axis by angle thetarad in radians.
   // The rotation axis is located at (x0,y0) in map coordinates.  There
   // are no assumed bounds on phirad.  Regions of the image that move
   // outside the map range are clipped and lost.  Regions of the image
   // that move into the range from outside are zero.  A new Map2D object
   // is created to hold the result, and the original object is untouched.
   // The caller must delete the returned object.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   Double_t cosphi = cos(phirad);
   Double_t sinphi = sin(phirad);

   Map2D *dst = new Map2D(*this);
   dst->Reset();
   for (Int_t i=1; i <= nxbins; ++i) {
      Double_t x = xmin+(i-0.5)*dx - x0;
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t y = ymin+(j-0.5)*dy - y0;
         Double_t xsrc = x*cosphi+y*sinphi + x0;
         Double_t ysrc = y*cosphi-x*sinphi + y0;
         Double_t xbins = (xsrc-xmin)/dx;
         Double_t ybins = (ysrc-ymin)/dy;
         if (xbins < 0 || xbins > nxbins ||
             ybins < 0 || ybins > nybins)
         {
            continue;
         }
         Int_t i0 = floor(xbins+0.5);
         Int_t j0 = floor(ybins+0.5);
         i0 = (i0 < 1)? 1 : i0;
         j0 = (j0 < 1)? 1 : j0;
         Int_t i1 = (i0 >= nxbins)? i0 : i0+1;
         Int_t j1 = (j0 >= nybins)? j0 : j0+1;
         Double_t fx1 = xbins+0.5-i0;
         Double_t fy1 = ybins+0.5-j0;
         Double_t fx0 = 1-fx1;
         Double_t fy0 = 1-fy1;
         Double_t z00 = GetBinContent(i0,j0);
         Double_t z10 = GetBinContent(i1,j0);
         Double_t z01 = GetBinContent(i0,j1);
         Double_t z11 = GetBinContent(i1,j1);
         Double_t zint0 = (z00 == 0)? ((fx0 > 0.5)? 0 : z10) :
                          (z10 == 0)? ((fx1 > 0.5)? 0 : z00) :
                          z00*fx0 + z10*fx1;
         Double_t zint1 = (z01 == 0)? ((fx0 > 0.5)? 0 : z11) :
                          (z11 == 0)? ((fx1 > 0.5)? 0 : z01) :
                          z01*fx0 + z11*fx1;
         Double_t zint = (zint0 == 0)? ((fy0 > 0.5)? 0 : zint1) :
                         (zint1 == 0)? ((fy1 > 0.5)? 0 : zint0) :
                         zint0*fy0 + zint1*fy1;
         dst->SetBinContent(i,j,zint);
      }
   }
   return dst;
}

Map2D &Map2D::Tilt(Double_t dzdx, Double_t dzdy, Double_t x0, Double_t y0)
{
   // Applies a linear transformation to the map image such that
   //     z'(x,y) = z(x,y) + dzdx (x-x0) + dzdy (y-y0)
   // Defaults for x0 and y0 are the coordinates of the image center.
   // This is equivalent to a rotation about an axis in the direction
   // omega = (-dzdy, dzdx, 0) in the limit of small |omega|, but it
   // is often simpler to find the values dzdx and dzdy than it is to
   // determine their equivalent Euler angles. Beyond leading order,
   // the two are not equivalent.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   if (x0 == DBL_MAX)
      x0 = (xmax + xmin)/2;
   if (y0 == DBL_MAX)
      y0 = (ymax + ymin)/2;
   for (Int_t i=1; i <= nxbins; ++i) {
      Double_t x = xmin+(i-0.5)*dx - x0;
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t y = ymin+(j-0.5)*dy - y0;
         Double_t z = GetBinContent(i,j);
         if (z != 0)
            SetBinContent(i,j,z+dzdx*(x-x0)+dzdy*(y-y0));
      }
   }
   return *this;
}

Map2D &Map2D::Rescale(Double_t sx, Double_t sy, Double_t sz)
{
   // Rescales the map image along all three axes by independent scale
   // factors.  The origin of the rescaling is taken to be the center
   // of the map in x and y, and z=0.  The original map is overwritten.
   // Note: the special case of sx=-1,sy=1,sz=-1 or sx=1,sy=-1,sz=-1
   // can be used to invert the surface, effectively flipping it over.

   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Map2D *tmp = rescale(sx,sy,sz,(xmax+xmin)/2,(ymax+ymin)/2,0);
   *this = *tmp;
   delete tmp;
   return *this;
}

Map2D &Map2D::Resize(Double_t xlow, Double_t ylow,
                     Double_t xhigh, Double_t yhigh, Int_t pixels) const
{
   // Creates a new map with bounds set by corner coordinates (xlow,ylow)
   // and (xhigh,yhigh), then copies this map into the new one.  The
   // original map (*this) is not modified.  The name of the new map is
   // the name of this map with the suffix "_rN", where N is an integer
   // starting with 1 and counting forward.

   std::string name = GetName();
   size_t sufi = name.rfind("_r");
   for (int revno=1; revno < 99999999; ++revno) {
      char str[99];
      sprintf(str,"_r%d",revno);
      name = name.substr(0,sufi) + str;
      if (gDirectory->FindObject(name.c_str()) == 0) {
         break;
      }
      sufi = name.rfind("_r");
   }
#if VERBOSE
   std::cout << "New map created with name " << name << std::endl;
#endif

   Double_t deltaX = GetXaxis()->GetBinWidth(1);
   Double_t deltaY = GetYaxis()->GetBinWidth(1);
   Int_t nxbins,nybins;
   if (pixels) {
      nxbins = (int)(xhigh-xlow)+1;
      nybins = (int)(yhigh-ylow)+1;
      xlow = GetXaxis()->GetXmin()+(xlow-1)*deltaX;
      ylow = GetYaxis()->GetXmin()+(ylow-1)*deltaY;
   }
   else {
      nxbins = (int)((xhigh-xlow)/deltaX + 0.999);
      nybins = (int)((yhigh-ylow)/deltaY + 0.999);
   }
   xhigh = xlow+nxbins*deltaX;
   yhigh = ylow+nybins*deltaY;
   TH2D hnew(name.c_str(),GetTitle(),nxbins,xlow,xhigh,nybins,ylow,yhigh);
   Map2D *newmap = new Map2D(hnew);
   newmap->Fill(this);
   return *newmap;
}

Map2D *Map2D::rescale(Double_t sx, Double_t sy, Double_t sz,
                      Double_t x0, Double_t y0, Double_t z0) const
{
   // Rescales the map image along all three axes by independent scale
   // factors.  The origin of the rescaling is (x0,y0) in map coordinates,
   // and z0 in height units.  A new object is created to hold the result
   // and the original object is untouched.  The caller must delete the
   // returned object.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;

   Map2D *dst = new Map2D(*this);
   dst->Reset();
   for (Int_t i=1; i <= nxbins; ++i) {
      Double_t x = (xmin+(i-0.5)*dx-x0)/sx + x0;
      for (Int_t j=1; j <=nybins; ++j) {
         Double_t y = (ymin+(j-0.5)*dy-y0)/sy + y0;
         Double_t xbins = (x-xmin)/dx;
         Double_t ybins = (y-ymin)/dy;
         Int_t i0 = floor(xbins+0.5);
         Int_t j0 = floor(ybins+0.5);
         i0 = (i0 < 1)? 1 : (i0 > nxbins)? nxbins : i0;
         j0 = (j0 < 1)? 1 : (j0 > nybins)? nybins : j0;
         Int_t i1 = (i0 >= nxbins)? i0 : i0+1;
         Int_t j1 = (j0 >= nybins)? j0 : j0+1;
         Double_t fx1 = xbins+0.5-i0;
         Double_t fy1 = ybins+0.5-j0;
         Double_t fx0 = 1-fx1;
         Double_t fy0 = 1-fy1;
         Double_t z00 = GetBinContent(i0,j0);
         Double_t z10 = GetBinContent(i1,j0);
         Double_t z01 = GetBinContent(i0,j1);
         Double_t z11 = GetBinContent(i1,j1);
         Double_t zint0 = (z00 == 0)? ((fx0 > 0.5)? 0 : z10) :
                          (z10 == 0)? ((fx1 > 0.5)? 0 : z00) :
                          z00*fx0 + z10*fx1;
         Double_t zint1 = (z01 == 0)? ((fx0 > 0.5)? 0 : z11) :
                          (z11 == 0)? ((fx1 > 0.5)? 0 : z01) :
                          z01*fx0 + z11*fx1;
         Double_t zint = (zint0 == 0)? ((fy0 > 0.5)? 0 : zint1) :
                         (zint1 == 0)? ((fy1 > 0.5)? 0 : zint0) :
                         zint0*fy0 + zint1*fy1;
         dst->SetBinContent(i,j,(zint-z0)*sz+z0);
      }
   }
   return dst;
}

Double_t Map2D::Correlation(const Map2D *map, Double_t contrast) const
{
   // Computes the height correlation with another map, useful for
   // aligning two images of the same terrain.  The contrast
   // argument serves as a weight factor on terms in the sum
   // where one or the other of the two maps is zero.  This allows
   // control over how zero regions are treated, such that
   //        contrast=0 : ignore zero regions
   //        contrast=1 : treat them like plateaus with z=0
   //   contrast=BIG : boundaries dominate, maps are B/W
   // Agreement between their coordinate ranges is not required,
   // but the two maps must have the same dimensions. 

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   if (map->GetXaxis()->GetNbins() != nxbins ||
       map->GetYaxis()->GetNbins() != nybins) {
      std::cerr << "Error in Map2D::Correlation - "
                << "maps must have the same dimensions."
                << std::endl;
      return 0;
   }
   Double_t sum00=0;
   Double_t sum10=0;
   Double_t sum01=0;
   Double_t sum11=0;
   Double_t sum20=0;
   Double_t sum02=0;
   for (Int_t i=1; i <= nxbins; ++i) {
      for (Int_t j=1; j <= nybins; ++j) {
         Double_t z0 = GetBinContent(i,j);
         Double_t z1 = map->GetBinContent(i,j);
         Double_t wgt = (z0*z1 == 0)? contrast : 1;
         if (wgt > 1) {
            z0 = (z0==0)? -wgt : z0;
            z1 = (z1==0)? -wgt : z1;
            wgt = 1;
         }
         sum00 += wgt;
         sum10 += wgt*z0;
         sum01 += wgt*z1;
         sum11 += wgt*z0*z1;
         sum20 += wgt*z0*z0;
         sum02 += wgt*z1*z1;
      }
   }
   Double_t covar = sum11/sum00-(sum10/sum00)*(sum01/sum00);
   Double_t var0 = sum20/sum00-(sum10/sum00)*(sum10/sum00);
   Double_t var1 = sum02/sum00-(sum01/sum00)*(sum01/sum00);
   return covar/sqrt((var0*var1 > 0)? var0*var1 : 1e-100);
}

Map2D &Map2D::Fill(const Map2D *map)
{
   // Copies the map from the source Map2D object into this object,
   // overwriting this object's data but preserving its dimensions and
   // axis limits.  Matte pixels in the source map are ignored, making
   // this method an effective way to overlay two non-overlapping images.
   // No new objects are created. Care is taken when changing the image
   // resolution from source to destination maps not to create aliasing
   // artifacts.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   Int_t nxbins_s = map->GetXaxis()->GetNbins();
   Int_t nybins_s = map->GetYaxis()->GetNbins();
   Double_t xmin_s = map->GetXaxis()->GetXmin();
   Double_t xmax_s = map->GetXaxis()->GetXmax();
   Double_t ymin_s = map->GetYaxis()->GetXmin();
   Double_t ymax_s = map->GetYaxis()->GetXmax();
   Double_t dx_s = (xmax_s-xmin_s)/nxbins_s;
   Double_t dy_s = (ymax_s-ymin_s)/nybins_s;

   Int_t pwritten=0;
   if (dx * dy < dx_s * dy_s) {

      // output map is higher resolution so interpolate the input map
 
      for (Int_t i=1; i <= nxbins; ++i) {
         Double_t x = xmin+(i-0.5)*dx;
         for (Int_t j=1; j <= nybins; ++j) {
            Double_t y = ymin+(j-0.5)*dy;
            Double_t xbin_s = (x-xmin_s)/dx_s;
            Double_t ybin_s = (y-ymin_s)/dy_s;
            Int_t i0 = floor(xbin_s+0.5);
            Int_t j0 = floor(ybin_s+0.5);
            if (i0 >= 1 && i0 <= nxbins_s && j0 >= 1 && j0 <= nybins_s) {
               Int_t i1 = (i0 == nxbins_s)? i0 : i0+1;
               Int_t j1 = (j0 == nybins_s)? j0 : j0+1;
               Double_t fx1 = xbin_s+0.5-i0;
               Double_t fy1 = ybin_s+0.5-j0;
               Double_t fx0 = 1-fx1;
               Double_t fy0 = 1-fy1;
               Double_t z00 = map->GetBinContent(i0,j0);
               Double_t z10 = map->GetBinContent(i1,j0);
               Double_t z01 = map->GetBinContent(i0,j1);
               Double_t z11 = map->GetBinContent(i1,j1);
               Double_t zint0 = (z00 == 0)? ((fx0 > 0.5)? 0 : z10) :
                                (z10 == 0)? ((fx1 > 0.5)? 0 : z00) :
                                z00*fx0 + z10*fx1;
               Double_t zint1 = (z01 == 0)? ((fx0 > 0.5)? 0 : z11) :
                                (z11 == 0)? ((fx1 > 0.5)? 0 : z01) :
                                z01*fx0 + z11*fx1;
               Double_t zint = (zint0 == 0)? ((fy0 > 0.5)? 0 : zint1) :
                               (zint1 == 0)? ((fy1 > 0.5)? 0 : zint0) :
                               zint0*fy0 + zint1*fy1;
               if (zint) {
                  SetBinContent(i,j,zint);
                  ++pwritten;
               }
            }
         }
      }
   }

   else {

      // output map is lower resolution so oversample the input map

      Int_t noversample = 3 * dx * dy / (dx_s * dy_s);
      Double_t xyrandom[2 * noversample];
      if (randoms == 0) {
         randoms = new TRandom3(0);
      }
      for (Int_t i=1; i <= nxbins; ++i) {
         Double_t x0 = xmin+(i-0.5)*dx;
         for (Int_t j=1; j <= nybins; ++j) {
            Double_t y0 = ymin+(j-0.5)*dy;
            randoms->RndmArray(2 * noversample, xyrandom);
            int nsamples = 0;
            Double_t zsum = 0;
            for (Int_t is=0; is < noversample; ++is) {
               Double_t x = x0 + dx*(xyrandom[2*is]-0.5);
               Double_t y = y0 + dy*(xyrandom[2*is+1]-0.5);
               Double_t xbin_s = (x-xmin_s)/dx_s;
               Double_t ybin_s = (y-ymin_s)/dy_s;
               Int_t i0 = floor(xbin_s+0.5);
               Int_t j0 = floor(ybin_s+0.5);
               if (i0 >= 1 && i0 <= nxbins_s && j0 >= 1 && j0 <= nybins_s) {
                  Int_t i1 = (i0 >= nxbins_s)? i0 : i0+1;
                  Int_t j1 = (j0 >= nybins_s)? j0 : j0+1;
                  Double_t fx1 = xbin_s+0.5-i0;
                  Double_t fy1 = ybin_s+0.5-j0;
                  Double_t fx0 = 1-fx1;
                  Double_t fy0 = 1-fy1;
                  Double_t z00 = map->GetBinContent(i0,j0);
                  Double_t z10 = map->GetBinContent(i1,j0);
                  Double_t z01 = map->GetBinContent(i0,j1);
                  Double_t z11 = map->GetBinContent(i1,j1);
                  Double_t zint0 = (z00 == 0)? ((fx0 > 0.5)? 0 : z10) :
                                   (z10 == 0)? ((fx1 > 0.5)? 0 : z00) :
                                   z00*fx0 + z10*fx1;
                  Double_t zint1 = (z01 == 0)? ((fx0 > 0.5)? 0 : z11) :
                                   (z11 == 0)? ((fx1 > 0.5)? 0 : z01) :
                                   z01*fx0 + z11*fx1;
                  Double_t zint = (zint0 == 0)? ((fy0 > 0.5)? 0 : zint1) :
                                  (zint1 == 0)? ((fy1 > 0.5)? 0 : zint0) :
                                  zint0*fy0 + zint1*fy1;
                  if (zint) {
                     zsum += zint;
                     ++nsamples;
                  }
               }
            }
            if (nsamples > noversample*0.85) {
               SetBinContent(i,j,zsum/nsamples);
               ++pwritten;
            }
         }
      }
   }
#if VERBOSE
   std::cout << "Map2D::Fill overlayed " << pwritten
             << " pixels." << std::endl;
#endif
   return *this;
}

Map2D &Map2D::Flood(Double_t value,
                    Double_t xlow, Double_t ylow,
                    Double_t xhigh, Double_t yhigh)
{
   // Sets all of the pixels within a rectangular region of the map
   // to a constant value, overwriting this object's data but preserving
   // its dimensions and axis limits.  Matte pixels in the source map are
   // overwritten as well as non-matte pixels.

   Int_t pwritten = 0;
   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   for (Int_t ix=1; ix <= nxbins; ++ix) {
      for (Int_t iy=1; iy <=nybins; ++iy) {
          double x = GetXaxis()->GetBinCenter(ix);
          double y = GetYaxis()->GetBinCenter(iy);
          if (x < xlow || x > xhigh || y < ylow || y > yhigh) {
             continue;
          }
          SetBinContent(ix,iy,value);
          ++pwritten;
      }
   }
#if VERBOSE
   std::cout << "Map2D::Flood replaced " << pwritten
             << " pixels." << std::endl;
#endif
   return *this;
}

Map2D &Map2D::FillVoids()
{
   // Use an intelligent interpolation/extrapolation algorithm to fill in
   // all of the matte pixels in this map with values from nearby non-
   // matte pixels. This is done by performing a sequence of calls to
   // the Smear method with increasing sigma values, and assigning matte
   // cells to the values obtained with the smallest sigma for which the
   // result is convergent. Contents of non-matte pixels are not affected.

   int nxbins = GetNbinsX();
   int nybins = GetNbinsY();
   double dx = (GetXaxis()->GetXmax() - GetXaxis()->GetXmin()) / nxbins;
   double dy = (GetYaxis()->GetXmax() - GetYaxis()->GetXmin()) / nybins;
   double sigma = sqrt(dx*dx + dy*dy);

   Map2D *mask = copy("mask");
   mask->Flood(1).Mask(this);

   int voids = 1;
   while (voids) {
      Map2D *numerator = copy("numerator");
      numerator->smear(sigma, sigma);
      Map2D *denominator = copy("denominator");
      denominator->Flood(1).Mask(this);
      denominator->smear(sigma, sigma);
      voids = 0;
      for (int i=1; i <= nxbins; ++i) {
         for (int j=1; j <= nybins; ++j) {
            double mvalue = mask->GetBinContent(i, j);
            double dvalue = denominator->GetBinContent(i, j);
            if (mvalue == 0) {
               if (dvalue > 1e-6) {
                  double nvalue = numerator->GetBinContent(i, j);
                  SetBinContent(i, j, nvalue/dvalue);
               }
               else {
                  voids++;
               }
            }
         }
      }
      sigma *= 1.5;
      delete numerator;
      delete denominator;
   }
   delete mask;
   return *this;
}

Map2D &Map2D::Smear(double xsigma, double ysigma)
{
   // Smear the contents of this map with a Gaussian smearing function.
   // Pixels with the matte value are not used in the smearing operation.
   // All pixels are overwritten by this operation. The horizontal and
   // vertical Gaussian smearing sigmas are given by input arguments
   // xsigma and ysigma.

   Map2D *mask = copy("mask");
   for (int i=1; i <= GetNbinsX(); ++i) {
      for (int j=1; j <= GetNbinsY(); ++j) {
         double cont = mask->GetBinContent(i, j);
         mask->SetBinContent(i, j, (cont == 0)? 0 : 1);
      }
   }
   smear(xsigma, ysigma);
   mask->smear(xsigma, ysigma);
   Divide(mask);
   delete mask;
   return *this;
}

Map2D *Map2D::smear(double xsigma, double ysigma)
{
   // Smear the contents of this map with a Gaussian smearing
   // function. All pixels in the map are modified without regard
   // to any special meaning of the matte value. This is a protected
   // method used by the public Smear method.

   // create a map of the smearing function
 
   Map2D *spot = copy("spot");
   int nxbins = GetNbinsX();
   int nybins = GetNbinsY();
   double xwidth = GetXaxis()->GetXmax() - GetXaxis()->GetXmin();
   double ywidth = GetYaxis()->GetXmax() - GetYaxis()->GetXmin();
   double dx = xwidth / nxbins;
   double dy = ywidth / nybins;
   double sigma2_i = pow(xsigma / dx, 2);
   double sigma2_j = pow(ysigma / dy, 2);
   for (int i=0; i <= nxbins/2; ++i) {
      double di2 = i*i;
      for (int j=0; j <= nybins/2; ++j) {
         double dj2 = j*j;
         double w = exp(-0.5 * (di2/sigma2_i + dj2/sigma2_j));
         spot->SetBinContent(i+1, j+1, w);
         if (i > 0)
            spot->SetBinContent(nxbins+1-i, j+1, w);
         if (j > 0)
            spot->SetBinContent(i+1, nybins+1-j, w);
         if (i > 0 && j > 0)
            spot->SetBinContent(nxbins+1-i, nybins+1-j, w);
      }
   }

   // take the DFT of the smearing spot map
 
   Map2D *spotFFTr = spot->copy("spotFFTr");
   spot->FFT(spotFFTr,"RE R2C P");
   Map2D *spotFFTi = spot->copy("spotFFTi");
   spot->FFT(spotFFTi,"IM R2C P");

   // take the DFT of the input map

   Map2D *mapFFTr = copy("mapFFTr");
   this->FFT(mapFFTr,"RE R2C P");
   Map2D *mapFFTi = copy("mapFFTi");
   this->FFT(mapFFTi,"IM R2C P");

   // use the convolution theorem to input with the spot shape

   int dim[2] = {nxbins, nybins};
   TVirtualFFT *invFFT = TVirtualFFT::FFT(2, dim, "C2R");
   for (int ix = 1; ix <= nxbins; ++ix) {
      for (int iy = 1; iy <= nybins/2 + 1; ++iy) {
         double map_r = mapFFTr->GetBinContent(ix,iy);
         double map_i = mapFFTi->GetBinContent(ix,iy);
         double spot_r = spotFFTr->GetBinContent(ix,iy);
         double spot_i = spotFFTi->GetBinContent(ix,iy);
         double prod_r = map_r*spot_r - map_i*spot_i;
         double prod_i = map_r*spot_i + map_i*spot_r;
         int index[2] = {ix - 1, iy - 1};
         invFFT->SetPoint(index, prod_r, prod_i);
      }
   }
   invFFT->Transform();
   TH2::TransformHisto(invFFT, this, "Re");
   Rescale(1, 1, 1.0/(nxbins*nybins));
   delete invFFT;
   delete spot;
   delete spotFFTr;
   delete spotFFTi;
   delete mapFFTr;
   delete mapFFTi;
   return this;
}

Map2D &Map2D::Mask(const Map2D *map, Int_t neg, Double_t value)
{
   // Overlays the input map on the output, and masks (overwrites with matte)
   // any cells that are zero in the input map.  Cells in the input map
   // that are non-zero are overwritten in *this with the constant "value"
   // unless value=0, in which case those pixels are preserved.  If argument
   // "neg" is non-zero, the input map is treated as a negative.  Both input
   // and output maps must have the same dimensions.  No new objects are
   // created. 

   Int_t nxbins = map->GetXaxis()->GetNbins();
   Int_t nybins = map->GetYaxis()->GetNbins();
   if (GetXaxis()->GetNbins() != nxbins || GetYaxis()->GetNbins() != nybins) {
      std::cerr << "Error in Map2D::Mask - argument map has different"
                << " pixel dimensions from the object being masked."
                << std::endl;
      return *this;
   }
   for (Int_t i=1; i <= nxbins; ++i) {
      for (Int_t j=1; j <= nybins; ++j) {
         double cont = map->GetBinContent(i,j);
         if ((!neg && cont == 0) || (neg && cont != 0)) {
            SetBinContent(i,j,0);
         }
         else if (value) {
            SetBinContent(i,j,value);
         }
      }
   }
   return *this;
}

Int_t Map2D::ZeroSuppress(Double_t threshold, Double_t zero)
{
   // Scans through the map and finds any pixels less than the supplied
   // threshold value, replacing them with the value zero. Matte pixels
   // are not touched.

   int count=0;
   for (int ix=1; ix <= GetXaxis()->GetNbins(); ++ix) {
      for (int iy=1; iy <= GetYaxis()->GetNbins(); ++iy) {
         double a = GetBinContent(ix,iy);
         if (a != 0 && a < threshold) {
            SetBinContent(ix,iy,zero);
            ++count;
         }
      }
   }
   return count;
}

Int_t Map2D::Despeckle(Int_t maxsize)
{
   // Searches the map for regions.  Pixels belong to the same region if
   // an unbroken path of contiguous pixels can be found that joins them.
   // Two pixels are contiguous if they share a common edge; touching at the
   // corners does not count.  Regions up to maxsize pixels are erased
   // (overwritten with matte) and a count of the remaining islands of greater
   // than maxsize pixels is returned.  If called with maxsize=0 then the
   // total number of islands is returned, and the map is not modified.

   std::vector<Int_t> rsize;
   std::vector<Int_t> rjoin;
   rsize.push_back(0);
   rjoin.push_back(0);
   Int_t regions=1;

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Int_t *mask = new Int_t[nxbins*nybins];
   Int_t region = 0;
   for (Int_t j=0; j < nybins; ++j) {
      if (GetBinContent(1,j+1) == 0) {
         mask[j] = region = 0;
         ++rsize[0];
      }
      else if (region > 0) {
         mask[j] = region;
         ++rsize[region];
      }
      else {
         mask[j] = region = regions++;
         rsize.push_back(1);
         rjoin.push_back(0);
      }
   }
   for (Int_t i=1; i < nxbins; ++i) {
      region = 0;
      Int_t j0 = i*nybins;
      for (Int_t j=0; j < nybins; ++j) {
         if (GetBinContent(i+1,j+1) == 0) {
            mask[j+j0] = region = 0;
            ++rsize[0];
            continue;
         }
         else if (mask[j+j0-nybins] != 0) {
            Int_t left = mask[j+j0-nybins];
            if (region == 0) {
               region = left;
            }
            else {
               while (rjoin[left] != 0) {
                  left = rjoin[left];
               }
               if (region > left) {
                  rjoin[region] = left;
                  rsize[left] += rsize[region];
                  rsize[region] = 0;
                  region = left;
               }
               else if (region < left) {
                  rjoin[left] = region;
                  rsize[region] += rsize[left];
                  rsize[left] = 0;
               }
            }
         }
         else if (region == 0) {
            region = regions++;
            rsize.push_back(0);
            rjoin.push_back(0);
         }
         mask[j+j0] = region;
         ++rsize[region];
      }
   }
   for (Int_t i=0; i < nxbins; ++i) {
      Int_t j0 = i*nybins;
      for (Int_t j=0; j < nybins; ++j) {
         region = mask[j+j0];
         int size = rsize[region];
         while (rjoin[region]) {
            region = rjoin[region];
            size += rsize[region];
         }
         if (size <= maxsize) {
            SetBinContent(i+1,j+1,0);
         }
      }
   }

   Int_t surviving = 0;
   for (Int_t r=1; r < regions; ++r) {
      if (rjoin[r] == 0 && rsize[r] > maxsize) {
         ++surviving;
      }
   }
   delete [] mask;
   return surviving;
}

Int_t Map2D::Despike(Double_t maxheight, Int_t regionsize)
{
   // Searches the map for localized spikes that appear in regions of
   // smoothly varying height, and replaces any spikes that are greater
   // than maxheight above or below the local average surface height with
   // the local neighborhood average.  The size of the neighborhood is
   // (2*regionsize+1)x(2*regionsize+1) in pixels.  This operation is not
   // reversible.

   Int_t nxbins = GetXaxis()->GetNbins()+1;
   Int_t nybins = GetYaxis()->GetNbins()+1;
   Double_t *zsum = new Double_t[nxbins*nybins];
   Int_t *nsum = new Int_t[nxbins*nybins];
   Int_t *mask = new Int_t[nxbins*nybins];
   for (Int_t ix=0; ix < nxbins; ++ix) {
      mask[ix] = 0;
      zsum[ix] = 0;
      nsum[ix] = 0;
   }

   for (Int_t iy=1; iy < nybins; ++iy) {
      Int_t rowsum0[nxbins];
      Double_t rowsum1[nxbins];
      Int_t sum0 = rowsum0[0] = 0;
      Double_t sum1 = rowsum1[0] = 0;
      for (Int_t ix=1; ix < nxbins; ++ix) {
        Double_t content = GetBinContent(ix,iy);
        mask[iy*nxbins+ix] = (content > 0)? 1 : 0;
        rowsum0[ix] = (content > 0)? ++sum0 : sum0;
        rowsum1[ix] = sum1 += content;
      }
      zsum[iy*nxbins] = 0;
      nsum[iy*nxbins] = 0;
      for (Int_t ix=1; ix < nxbins; ++ix) {
        Int_t start = ix-regionsize;
        Int_t stop = ix+regionsize;
        start = (start < 1)? 0 : start-1;
        stop = (stop >= nxbins)? nxbins-1 : stop;
        nsum[iy*nxbins+ix] = rowsum0[stop] - rowsum0[start];
        zsum[iy*nxbins+ix] = rowsum1[stop] - rowsum1[start];
      }
   }
   for (Int_t ix=1; ix < nxbins; ++ix) {
      Int_t colsum0[nybins];
      Double_t colsum1[nybins];
      Int_t sum0 = colsum0[0] = 0;
      Double_t sum1 = colsum1[0] = 0;
      for (Int_t iy=1; iy < nybins; ++iy) {
        colsum0[iy] = sum0 += nsum[iy*nxbins+ix];
        colsum1[iy] = sum1 += zsum[iy*nxbins+ix];
      }
      for (Int_t iy=1; iy < nybins; ++iy) {
        Int_t start = iy-regionsize;
        Int_t stop = iy+regionsize;
        start = (start < 1)? 0 : start-1;
        stop = (stop >= nybins)? nybins-1 : stop;
        nsum[iy*nxbins+ix] = colsum0[stop] - colsum0[start];
        zsum[iy*nxbins+ix] = colsum1[stop] - colsum1[start];
      }
   }

   // mask(ix,iy) encodes the status of pixel ix,iy as follows:
   //   0 : matte pixel, ignore.
   //   1 : pixel has height data, include in local neighborhood average.
   //  -1 : pixel has height data, but exclude from local neighborhood average
   //       and mark as a spike to be overwritten later.

   Int_t changes=1;
   while (changes) {
      changes = 0;
      for (Int_t ix=1; ix < nxbins; ++ix) {
         for (Int_t iy=1; iy < nybins; ++iy) {
            if (mask[iy*nxbins+ix] == 1) {
               Double_t z = GetBinContent(ix,iy);
               Double_t zave = zsum[iy*nxbins+ix]/nsum[iy*nxbins+ix];
               if (fabs(z-zave) > maxheight) {
                  Int_t rx0 = ix-regionsize;
                  Int_t rx1 = ix+regionsize;
                  Int_t ry0 = iy-regionsize;
                  Int_t ry1 = iy+regionsize;
                  rx0 = (rx0 < 1)? 1 : rx0;
                  ry0 = (ry0 < 1)? 1 : ry0;
                  rx1 = (rx1 < nxbins)? rx1 : nxbins-1;
                  ry1 = (ry1 < nybins)? ry1 : nybins-1;
                  for (Int_t rx=rx0; rx <= rx1; ++rx) {
                     for (Int_t ry=ry0; ry <= ry1; ++ry) {
                       zsum[ry*nxbins+rx] -= z;
                       nsum[ry*nxbins+rx] -= 1;
                     }
                  }
                  mask[iy*nxbins+ix] = -1;
                  ++changes;
               }
            }
         }
      }
   }

   Int_t spikes = 0;
   for (Int_t ix=1; ix < nxbins; ++ix) {
      for (Int_t iy=1; iy < nybins; ++iy) {
         Int_t mask_ = mask[iy*nxbins+ix];
         Int_t nsum_ = nsum[iy*nxbins+ix];
         if (mask_ == 1) {
            SetBinContent(ix,iy,GetBinContent(ix,iy));
         }
         else if (mask_ == -1 && nsum_ > 0) {
            SetBinContent(ix,iy,zsum[iy*nxbins+ix]/nsum_);
            ++spikes;
         }
      }
   }

   delete [] zsum;
   delete [] nsum;
   delete [] mask;
   return spikes;
}

Map2D &Map2D::Center()
{
   // Finds the center of gravity of the non-zero portion of the map
   // and applies a shift in x,y to locate the center at the midpoint.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;

   Double_t sum=0;
   Double_t sumx=0;
   Double_t sumy=0;
   for (Int_t ix=1; ix < nxbins; ++ix) {
      for (Int_t iy=1; iy < nybins; ++iy) {
         if (GetBinContent(ix,iy) == 0) {
            continue;
         }
         sumx += ix;
         sumy += iy;
         ++sum;
      }
   }
   if (sum == 0) {
      return *this;
   }
   Double_t xshift = ((nxbins+1)/2 - sumx/sum) * dx;
   Double_t yshift = ((nybins+1)/2 - sumy/sum) * dy;
   return Shift(xshift,yshift);
}

Map2D &Map2D::Level()
{
   // Locates the left-most and the right-most pixels in the map with
   // non-zero values and applies an in-plane rotation such that those
   // two pixels are in the same row (y-value) of the map.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;

   Int_t left[3] = {nxbins,0,0};
   Int_t right[3] = {0,0,0};
   for (Int_t ix=1; ix <= nxbins; ++ix) {
      for (Int_t iy=1; iy <= nybins; ++iy) {
         double cont = GetBinContent(ix,iy);
         if (cont == 0) {
            continue;
         }
         if (ix < left[0]) {
            left[0] = ix;
            left[1] = iy;
            left[2] = 1;
         }
         else if (ix == left[0]) {
            left[1] += iy;
            left[2] += 1;
         }
         else if (ix > right[0]) {
            right[0] = ix;
            right[1] = iy;
            right[2] = 1;
         }
         else if (ix == right[0]) {
            right[1] += iy;
            right[2] += 1;
         }
      }
   }

   Double_t alpha = atan2((right[1]/right[2] - left[1]/left[2]) * dy,
                          (right[0] - left[0]) * dx);
   return Rotate(0,0,-alpha);
}

Map2D &Map2D::Normalize()
{
   // Finds the maximum and minimum values of the map, excluding cells
   // with zero value, and sets the Max and Min to the corresponding values.

   double max = -1e300;
   double min = +1e300;
   for (int ix=1; ix <= GetNbinsX(); ++ix) {
      for (int iy=1; iy <= GetNbinsY(); ++iy) {
         double cont = GetBinContent(ix,iy);
         if (cont == 0) {
            continue;
         }
         max = (cont > max)? cont : max;
         min = (cont < min)? cont : min;
      }
   }
   SetMaximum(max);
   SetMinimum(min);
   return *this;
}

Double_t Map2D::TotalCurl(const Map2D *Vx, const Map2D *Vy,
                          Map2D **silhouette)
{
   // Forms a closed path around the outermost periphery of the map, and
   // computes the total curl of a 2D vector field within that path using
   // Stoke's theorem.  The two components of the vector field within the
   // plane of the map are accepted as arguments Vx,Vy.  The result is
   // returned in the units of V times coordinate length.  The two maps
   // given as arguments must have the same dimensions and bounds in x and 
   // y.  The object being used to compute the curl is overwritten with a
   // map filled with zeros everywhere except on the contour around which
   // the Stokes integral was performed, with each cell on the contour
   // containing the running integral value up to that point.  If optional
   // argument silhouette is provided by the caller, it is overwritten upon
   // return with a pointer to a new Map2D object with the same shape as
   // Vx,Vy whose contents represent an edge-enhanced silhouette of the
   // overlaid input maps, with bin values defined as follows. 
   //        0    : the bin is outside the contour,
   //      (0,1)  : the bin is inside the contour,
   //     [1,inf) : the bin is on the contour.
   // It is up to the user to delete the silhouette object when it is no
   // longer needed.

   Int_t nxbins = Vx->GetXaxis()->GetNbins();
   Int_t nybins = Vx->GetYaxis()->GetNbins();
   Double_t xmin = Vx->GetXaxis()->GetXmin();
   Double_t xmax = Vx->GetXaxis()->GetXmax();
   Double_t ymin = Vx->GetYaxis()->GetXmin();
   Double_t ymax = Vx->GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   if (Vy->GetXaxis()->GetNbins() != nxbins ||
       Vy->GetYaxis()->GetNbins() != nybins ||
       Vy->GetXaxis()->GetXmin() != xmin ||
       Vy->GetXaxis()->GetXmax() != xmax ||
       Vy->GetYaxis()->GetXmin() != ymin ||
       Vy->GetYaxis()->GetXmax() != ymax) {
      std::cerr << "Error in Map2D::TotalCurl - the two input maps have"
                << " different pixel dimensions or axis bounds."
                << std::endl;
      return 0;
   }

   clone(Vx);
   Mask(Vy);
   int specksize=10;
   while (Despeckle(specksize) > 1) {
      specksize *= 2;
   }
   if (Despeckle() != 1) {
      std::cerr << "Error in Map2D::TotalCurl - the map is too fragmented"
                << " for this algorithm, please despeckle and try again."
                << std::endl;
      return 0;
   }

   // Highlight the boundary pixels
   Double_t *bound = new Double_t[nxbins*nybins];
   for (Int_t i=1; i <= nxbins; ++i) {
      Int_t j0 = (i-1)*nybins-1;
      for (Int_t j=1; j <= nybins; ++j) {
         if (GetBinContent(i,j) != 0) {
            Double_t south = (j == 1)?      0 : GetBinContent(i,j-1);
            Double_t north = (j == nybins)? 0 : GetBinContent(i,j+1);
            Double_t west  = (i == 1)?      0 : GetBinContent(i-1,j);
            Double_t east  = (i == nxbins)? 0 : GetBinContent(i+1,j);
            bound[j+j0] = 4.1 - ((south == 0)? 0 : 1) - ((north == 0)? 0 : 1)
                              - ((west == 0)?  0 : 1) - ((east == 0)?  0 : 1);
         }
         else {
            bound[j+j0] = 0;
         }
      }
   }
   for (Int_t i=1; i <= nxbins; ++i) {
      Int_t j0 = (i-1)*nybins-1;
      for (Int_t j=1; j <= nybins; ++j) {
         SetBinContent(i,j,bound[j+j0]);
      }
   }

   // Direction codes are: 0=east, 1=north, 2=west, 3=south
   Int_t di[4] = {+1,0,-1,0};
   Int_t dj[4] = {0,+1,0,-1};
   Int_t back[4] = {2,3,0,1};

   // Start at the center of the east coast and head northeast
   Int_t istep=0;
   Int_t jstep=0;
   for (Int_t i=nxbins/2,j=nybins/2; i < nxbins; ++i) {
      if (bound[i*nybins+j] > 1) {
         istep = i;
         jstep = j;
      }
   }
   Int_t istart=istep;
   Int_t jstart=jstep;
   Int_t stepdir=2;

   // Dig an outer perimeter loop around the primary region
   std::list<Int_t> dirseq;
   Int_t steps=0;
#ifdef MANUAL_STEPPING_IN_TOTALCURL
   Int_t skip=0;
   TCanvas *c1=0;
#endif
   while (1 > 0) {
      SetBinContent(istep+1,jstep+1,5.1);

#ifdef MANUAL_STEPPING_IN_TOTALCURL
      if (skip-- == 0) {
         GetXaxis()->SetRangeUser(xmin+(istep-25)*dx,xmin+(istep+25)*dx);
         GetYaxis()->SetRangeUser(ymin+(jstep-25)*dy,ymin+(jstep+25)*dy);
         if (c1 == 0) {
            gDirectory->GetObject("c1",c1);
            if (c1 == 0) {
               c1 = new TCanvas("c1","c1",500,500);
            }
            c1->cd();
         }
         Draw("colz");
         c1->Update();
         std::cout << "Enter number of steps to skip before pausing next: ";
         std::cin >> skip;
         if (skip < 0) {
            return 0;
         }
      }
#endif

      Double_t score = 4;
      Int_t nextdir = -1;
      for (Int_t d=1; d < 4; ++d) {
         Int_t dir = back[(stepdir+d)%4];
         Double_t s = bound[(istep+di[dir])*nybins+jstep+dj[dir]];
         if (s > 1 && s < score) {
            nextdir = dir;
            score = s;
         }
      }
      if (score > 3) {
         for (Int_t d=1; d < 4; ++d) {
            Int_t dir = back[(stepdir+d)%4];
            Int_t itry = istep+di[dir];
            Int_t jtry = jstep+dj[dir];
            if (bound[itry*nybins+jtry] > 0) {
               Double_t sees=0;
               for (Int_t dtry=0; dtry < 4; ++dtry) {
                  Double_t s = bound[(itry+di[dtry])*nybins+jtry+dj[dtry]];
                  if (s > 1 && s < 3) {
                     ++sees;
                  }
               }
               if (sees > 0) {
                  nextdir = dir;
                  score = 3;
                  break;
               }
            }
         }
      }
      if (score > 3) {
         if (steps) {
            SetBinContent(istep+1,jstep+1,4.1);
            istep -= di[stepdir];
            jstep -= dj[stepdir];
            stepdir = dirseq.back();
            dirseq.pop_back();
            --steps;
            continue;
         }
         std::cerr << "Error in Map2D::TotalCurl - algorithm is stuck after "
                   << steps << " steps, and cannot continue."
                   << std::endl;
         return 0;
      }
      SetBinContent(istep+1,jstep+1,3.8);
      dirseq.push_back(stepdir);
      stepdir = nextdir;
      istep += di[stepdir];
      jstep += dj[stepdir];
      bound[istep*nybins+jstep] = -1;
      steps++;
      if (istep == istart && jstep == jstart) {
         dirseq.push_back(stepdir);
         break;
      }
   }

   // Fill in any voids in the interior

   for (Int_t ix=2; ix < nxbins; ++ix) {
      for (Int_t iy=2; iy < nybins; ++iy) {
         if (GetBinContent(ix,iy) < 2) {
            SetBinContent(ix,iy,0.1);
         }
      }
   }
   int voids=1;
   while (voids) {
      voids = 0;
      for (Int_t ix=2; ix < nxbins; ++ix) {
         for (Int_t iy=2; iy < nybins; ++iy) {
            double u = GetBinContent(ix,iy);
            if (u > 0 && u < 1 && (
                GetBinContent(ix,iy-1) == 0 ||
                GetBinContent(ix-1,iy) == 0 ))
            {
               SetBinContent(ix,iy,0);
               ++voids;
            }
         }
      }
      for (Int_t ix=nxbins-1; ix > 1; --ix) {
         for (Int_t iy=nybins-1; iy > 1; --iy) {
            double u = GetBinContent(ix,iy);
            if (u > 0 && u < 1 && (
                GetBinContent(ix,iy+1) == 0 ||
                GetBinContent(ix+1,iy) == 0 ))
            {
               SetBinContent(ix,iy,0);
               ++voids;
            }
         }
      }
   }

   // Save the silhouette map, if requested

   GetXaxis()->SetRange();
   GetYaxis()->SetRange();
   if (silhouette) {
      if (*silhouette) {
         delete *silhouette;
      }
      *silhouette = new Map2D(*this);
   }

   // Now follow the loop and compute the path integral

   TH1D *loopcurl = (TH1D*)gROOT->FindObject("loopcurl");
   TH1I *looppixi = (TH1I*)gROOT->FindObject("looppixi");
   TH1I *looppixj = (TH1I*)gROOT->FindObject("looppixj");
   if (loopcurl)
      delete loopcurl;
   if (looppixi)
      delete looppixi;
   if (looppixj)
      delete looppixj;
   loopcurl= new TH1D("loopcurl", "loop running curl",
                      dirseq.size(), 0, dirseq.size());
   looppixi = new TH1I("looppixi", "loop path i_pixel",
                       dirseq.size(), 0, dirseq.size());
   looppixj = new TH1I("looppixj", "loop path j_pixel",
                       dirseq.size(), 0, dirseq.size());

   Reset();
   istep = istart;
   jstep = jstart;
   Double_t curl=0;
   Int_t step=1;
   looppixi->SetBinContent(step,istep);
   looppixj->SetBinContent(step,jstep);
   loopcurl->SetBinContent(step,curl);
   std::list<Int_t>::iterator stepit = dirseq.begin();
   for (++stepit; stepit != dirseq.end(); ++stepit) {
      switch (*stepit) {
       case 0: //east
         curl += Vx->GetBinContent(istep+1,jstep+1)*dx;
         ++istep;
         break;
       case 1: //north
         curl += Vy->GetBinContent(istep+1,jstep+1)*dy;
         ++jstep;
         break;
       case 2: //west
         curl -= Vx->GetBinContent(istep,jstep+1)*dx;
         --istep;
         break;
       case 3: //south
         curl -= Vy->GetBinContent(istep+1,jstep)*dy;
         --jstep;
         break;
      }
      ++step;
      looppixi->SetBinContent(step,istep);
      looppixj->SetBinContent(step,jstep);
      loopcurl->SetBinContent(step,curl);
      SetBinContent(istep+1,jstep+1,curl);
   }
   if (istep != istart || jstep != jstart) {
      std::cerr << "Error in Map2D::TotalCurl - path around map periphery"
                << " is not closed, and cannot continue."
                << std::endl;
      std::cerr << "   istart,jstart=" << istart << "," << jstart << std::endl
                << "   istop,jstop=" << istep << "," << jstep << std::endl;
      return 0;
   }

   delete [] bound;
   return curl;
}

Map2D &Map2D::GetSilhouette()
{
   // Calls TotalCurl for the purpose of computing the silhouette of a map.
   // The contents of *this are overwritten with the silhouette.

   Map2D *silh;
   Map2D *me = copy("me");
   TotalCurl(me, me, &silh);
   Fill(silh);
   delete me;
   delete silh;
   return *this;
}

Double_t Map2D::Curl(const Map2D *Vx, const Map2D *Vy)
{
   // Computes the curl of the 2D vector field represented by the components
   // Vx,Vy and stores the result in the *this map.  The value stored in
   // each cell of the output map represents the integral of the curl within
   // that cell, so that the sum over all cells in the map should equal the
   // line integral of the field around a closed loop enclosing the non-zero
   // field region, according to Stoke's theorem.  The complementary method
   // TotalCurl() is provided to compute the same result using the loop line
   // integral.  The return value is the surface integral of the curl.

   Int_t nxbins = Vx->GetXaxis()->GetNbins();
   Int_t nybins = Vx->GetYaxis()->GetNbins();
   Double_t xmin = Vx->GetXaxis()->GetXmin();
   Double_t xmax = Vx->GetXaxis()->GetXmax();
   Double_t ymin = Vx->GetYaxis()->GetXmin();
   Double_t ymax = Vx->GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   if (Vy->GetXaxis()->GetNbins() != nxbins ||
       Vy->GetYaxis()->GetNbins() != nybins ||
       Vy->GetXaxis()->GetXmin() != xmin ||
       Vy->GetXaxis()->GetXmax() != xmax ||
       Vy->GetYaxis()->GetXmin() != ymin ||
       Vy->GetYaxis()->GetXmax() != ymax) {
      std::cerr << "Error in Map2D::Curl - the two input maps have"
                << " different pixel dimensions or axis bounds."
                << std::endl;
      return 0;
   }

   clone(Vx);
   Mask(Vy);
   int specksize=10;
   while (Despeckle(specksize) > 1) {
      specksize *= 2;
   }
   if (Despeckle() != 1) {
      std::cerr << "Error in Map2D::Curl - the map is too fragmented"
                << " for this algorithm, please despeckle and try again."
                << std::endl;
      return 0;
   }

   for (Int_t ix=2; ix < nxbins; ++ix) {
      for (Int_t iy=2; iy < nybins; ++iy) {
         SetBinContent(ix,iy,0);
      }
   }
   Double_t sum = 0;
   for (Int_t ix=2; ix < nxbins; ++ix) {
      for (Int_t iy=2; iy < nybins; ++iy) {
         double Vx_i_j = Vx->GetBinContent(ix,iy);
         double Vy_i_j = Vy->GetBinContent(ix,iy);
         double Vx_i_jp = Vx->GetBinContent(ix,iy+1);
         double Vy_i_jp = Vy->GetBinContent(ix,iy+1);
         double Vx_ip_j = Vx->GetBinContent(ix+1,iy);
         double Vy_ip_j = Vy->GetBinContent(ix+1,iy);
         double Vx_ip_jp = Vx->GetBinContent(ix+1,iy+1);
         double Vy_ip_jp = Vy->GetBinContent(ix+1,iy+1);
         if (Vx_i_j != 0 && Vx_i_jp != 0 && Vx_ip_j != 0 && Vx_ip_jp != 0 &&
             Vy_i_j != 0 && Vy_i_jp != 0 && Vy_ip_j != 0 && Vy_ip_jp != 0)
         {
            double curlint = (Vy_ip_j - Vy_i_j) * dy
                           - (Vx_i_jp - Vx_i_j) * dx;
            //SetBinContent(ix,iy,GetBinContent(ix,iy)-9);
            //SetBinContent(ix+1,iy,GetBinContent(ix+1,iy)+10);
            //SetBinContent(ix,iy+1,GetBinContent(ix,iy+1)-1);
            SetBinContent(ix,iy,curlint);
            sum += curlint;
         }
         else {
            SetBinContent(ix,iy,0);
         }
      }
   }
   return sum;
}

Map2D &Map2D::Uncurl(const Map2D *Vx, const Map2D *Vy, Int_t comp)
{
   // Computes a correction to a 2D vector field defined on a plane such
   // that the resulting field has zero curl.  This essentially replaces
   // one of the two degrees of freedom of the field with the constraint.
   // The argument "comp" tells which of the two components is replaced:
   //      comp = 0: return a replacement for Vx;
   //      comp = 1; return a replacement for Vy.
   // The read-only map arguments Vx,Vy are not modified.  
   //
   // Algorithm:
   // There are an infinite number of replacements for Vx,Vy such that
   // Del x (Vx,Vy) = 0, all of which are related to one another through
   // gauge transformations.  The purpose here is to modify the field in
   // as minimal a fashion as possible to eliminate the curl.  To simplify
   // the problem, consider only the subset of solutions that leaves Vx [Vy]
   // unchanged, as selected by the user.  This reduces the solution to
   // finding a set of independent columns [rows] that zeros the curl within
   // that column [row] subject to some criterion for minimal modification.
   // In 1D the zero-curl constrain reduces to a first-order differences
   // equation (first-order inhomogenous differential equation in 1D written
   // in finite-element form) which has just one overall undetermined offset.
   //  I chose the offset to minimize the sum of the squared differences
   // from the corresponding input map.  This uniquely determines the result.

   Int_t nxbins = GetXaxis()->GetNbins();
   Int_t nybins = GetYaxis()->GetNbins();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   if (Vx->GetXaxis()->GetNbins() != nxbins ||
       Vx->GetYaxis()->GetNbins() != nybins ||
       Vx->GetXaxis()->GetXmin() != xmin ||
       Vx->GetXaxis()->GetXmax() != xmax ||
       Vx->GetYaxis()->GetXmin() != ymin ||
       Vx->GetYaxis()->GetXmax() != ymax ||
       Vy->GetXaxis()->GetNbins() != nxbins ||
       Vy->GetYaxis()->GetNbins() != nybins ||
       Vy->GetXaxis()->GetXmin() != xmin ||
       Vy->GetXaxis()->GetXmax() != xmax ||
       Vy->GetYaxis()->GetXmin() != ymin ||
       Vy->GetYaxis()->GetXmax() != ymax) {
      std::cerr << "Error in Map2D::Uncurl - the input maps have"
                << " different pixel dimensions or axis bounds."
                << std::endl;
      return *this;
   }

   if (comp == 0) {
      for (Int_t ix=1; ix < nxbins; ++ix) {
         double sum0 = 0;
         double sum1 = 0;
         double Vxp[nybins];
         for (Int_t iy=1; iy < nybins; ++iy) {
            double Vx_i_j = Vx->GetBinContent(ix,iy);
            double Vy_i_j = Vy->GetBinContent(ix,iy);
            double Vx_i_jp = Vx->GetBinContent(ix,iy+1);
            double Vy_i_jp = Vy->GetBinContent(ix,iy+1);
            double Vx_ip_j = Vx->GetBinContent(ix+1,iy);
            double Vy_ip_j = Vy->GetBinContent(ix+1,iy);
            double Vx_ip_jp = Vx->GetBinContent(ix+1,iy+1);
            double Vy_ip_jp = Vy->GetBinContent(ix+1,iy+1);
            if (Vx_i_j != 0 && Vx_i_jp != 0 && Vx_ip_j != 0 && Vx_ip_jp != 0 &&
                Vy_i_j != 0 && Vy_i_jp != 0 && Vy_ip_j != 0 && Vy_ip_jp != 0)
            {
               if (Vxp[iy-1] == 0) {
                  Vxp[iy-1] = Vx_i_j;
                  ++sum0;
               }
               Vxp[iy] = Vxp[iy-1] + (Vy_ip_j - Vy_i_j) * dy/dx;
            }
            else {
               Vxp[iy] = 0;
            }
            sum1 += Vx_i_jp - Vxp[iy];
            ++sum0;
         }
         double offset = (sum0 > 0)? sum1/sum0 : 0;
         for (Int_t iy=1; iy <= nybins; ++iy) {
            if (Vxp[iy-1] != 0) {
               SetBinContent(ix,iy,Vxp[iy-1]+offset);
            }
            else {
               SetBinContent(ix,iy,0);
            }
         }
      }
   }
   else {
      for (Int_t iy=1; iy < nybins; ++iy) {
         double sum0 = 0;
         double sum1 = 0;
         double Vyp[nxbins];
         for (Int_t ix=1; ix < nxbins; ++ix) {
            double Vx_i_j = Vx->GetBinContent(ix,iy);
            double Vy_i_j = Vy->GetBinContent(ix,iy);
            double Vx_i_jp = Vx->GetBinContent(ix,iy+1);
            double Vy_i_jp = Vy->GetBinContent(ix,iy+1);
            double Vx_ip_j = Vx->GetBinContent(ix+1,iy);
            double Vy_ip_j = Vy->GetBinContent(ix+1,iy);
            double Vx_ip_jp = Vx->GetBinContent(ix+1,iy+1);
            double Vy_ip_jp = Vy->GetBinContent(ix+1,iy+1);
            if (Vx_i_j != 0 && Vx_i_jp != 0 && Vx_ip_j != 0 && Vx_ip_jp != 0 &&
                Vy_i_j != 0 && Vy_i_jp != 0 && Vy_ip_j != 0 && Vy_ip_jp != 0)
            {
               if (Vyp[ix-1] == 0) {
                  Vyp[ix-1] = Vy_i_j;
                  ++sum0;
               }
               Vyp[ix] = Vyp[ix-1] + (Vx_i_jp - Vx_i_j) * dx/dy;
               sum1 += Vy_ip_j - Vyp[ix];
               ++sum0;
            }
            else {
               Vyp[ix] = 0;
            }
         }
         double offset = (sum0 > 0)? sum1/sum0 : 0;
         for (Int_t ix=1; ix <= nxbins; ++ix) {
            if (Vyp[ix-1] != 0) {
               SetBinContent(ix,iy,Vyp[ix-1]+offset);
            }
            else {
               SetBinContent(ix,iy,0);
            }
         }
      }
   }
   return *this;
}

Double_t Map2D::Divergence(const Map2D *Vx, const Map2D *Vy)
{
   // Computes the divergence of the vector field V whose x,y components
   // are passed in as the maps Vx,Vy.  The contents of map *this are
   // overwritten with the result.  The return value is the total
   // divergence integrated over the output map, in units of V times
   // coordinate length.  The two maps given as arguments must have the
   // same dimensions and bounds in x and y.

   Int_t nxbins = Vx->GetXaxis()->GetNbins();
   Int_t nybins = Vx->GetYaxis()->GetNbins();
   Double_t xmin = Vx->GetXaxis()->GetXmin();
   Double_t xmax = Vx->GetXaxis()->GetXmax();
   Double_t ymin = Vx->GetYaxis()->GetXmin();
   Double_t ymax = Vx->GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   if (Vy->GetXaxis()->GetNbins() != nxbins ||
       Vy->GetYaxis()->GetNbins() != nybins ||
       Vy->GetXaxis()->GetXmin() != xmin ||
       Vy->GetXaxis()->GetXmax() != xmax ||
       Vy->GetYaxis()->GetXmin() != ymin ||
       Vy->GetYaxis()->GetXmax() != ymax) {
      std::cerr << "Error in Map2D::Divergence - the two input maps have"
                << " different pixel dimensions or axis bounds."
                << std::endl;
      return 0;
   }

   clone(Vx);
   Double_t divsum = 0;
   for (Int_t iy=1; iy <= GetNbinsY(); ++iy) {
      for (Int_t ix=1; ix <= GetNbinsX(); ++ix) {
         if (ix > 1 && iy > 1) {
            Double_t Vx1 = Vx->GetBinContent(ix,iy);
            Double_t Vx0 = Vx->GetBinContent(ix-1,iy);
            Double_t Vy1 = Vy->GetBinContent(ix,iy);
            Double_t Vy0 = Vy->GetBinContent(ix,iy-1);
            if (Vx1*Vx0*Vy0*Vy1 != 0) {
               Double_t div = (Vx1-Vx0)/dx + (Vy1-Vy0)/dy;
               SetBinContent(ix,iy,div);
               divsum += div;
               continue;
            }
         }
         SetBinContent(ix,iy,0);
      }
   }
   return divsum*dx*dy;
}

Int_t Map2D::Gradient(Map2D *Vx, Map2D *Vy) const
{
   // Computes the gradient of the scalar field V in map *this and
   // stores the x,y components of the gradient in output maps Vx,Vy.
   // Arguments Vx,Vy must point to maps which have been created by the
   // user at input, otherwise the corresponding gradient component is not
   // computed.  At output these maps are overwritten by the computed
   // gradient components.  If both pointer arguments are zero at input
   // the routine does nothing and returns 0, otherwise it returns the
   // total number of pixels in the map *this for which the gradient vector
   // has been computed.
   
   if (Vx == 0 && Vy == 0) {
      return 0;
   }
   if (Vx) {
      *Vx = *this;
      Vx->Reset();
   }
   if (Vy) {
      *Vy = *this;
      Vy->Reset();
   }

   Double_t dx = GetXaxis()->GetBinWidth(1);
   Double_t dy = GetYaxis()->GetBinWidth(1);
   Int_t pixels=0;
   for (Int_t iy=1; iy < GetNbinsY(); ++iy) {
      for (Int_t ix=1; ix < GetNbinsX(); ++ix) {
         Double_t h00 = GetBinContent(ix,iy);
         Int_t count = 0;
         if (Vx) {
            Double_t h10 = GetBinContent(ix+1,iy);
            Double_t val = (h10*h00 != 0)? (h10-h00)/dx : 0;
            Vx->SetBinContent(ix,iy,val);
            count += (val != 0)? 1 : 0;
         }
         if (Vy) {
            Double_t h01 = GetBinContent(ix,iy+1);
            Double_t val = (h01*h00 != 0)? (h01-h00)/dy : 0;
            Vy->SetBinContent(ix,iy,val);
            count += (val != 0)? 1 : 0;
         }
         pixels += (count == 0)? 0 : 1;
      }
   }
   return pixels;
}

Int_t Map2D::PoissonSolve(const Map2D *rho, const Map2D *mask)
{
   // Solves the Poisson equation in two dimensions with a source given by
   // input map rho, subject to the boundary conditions contained in this
   // object at entry.  If input argument "mask" points to an existing object,
   // that object is interpreted as a mask for the pixels in map *this that
   // contain boundary values, otherwise all non-zero pixels in map *this
   // at input are considered to specify boundary values.  The algorithm
   // will make use of as many (or few) boundary values as the user provides,
   // although the solution may not satisfy all of them exactly if the
   // problem is over-determined, i.e. too many values are provided, or they
   // are clustered in too compact a region of the map.  In any case, the
   // solution will attempt to match all of the provided data points as
   // closely as possible, subject to the constraints of Poisson's equation
   // and the resolution of the map.  All cells in the map *this are over
   // written with the solution except those specifying boundary conditions.
   //
   // Return value:
   // The return value is the rank of the matrix that expresses LaPlace's
   // equation for the given boundary conditions.  The rank expresses the
   // effective number of independent boundary values that were used in
   // attempting the satisfy the stated boundary conditions.
   //
   // Algorithm:
   // First the Green's function for the 2D Poisson equation with a point
   // (single pixel) source is evaluated for each non-zero pixel in rho,
   // and the results are superimposed.  At this point, the solution obeys
   // Poisson's equation but with arbitrary boundary conditions. The map
   // is now subtracted from map *this that specifies the boundary values
   // for the problem, and the resulting map is handed to LaplaceSolve()
   // to find the homogeneous part of the solution.  Adding the homogenous
   // and inhomogenous parts yields the full solution, which is returned
   // in this object.
   //
   // Timing:
   // The time required to complete this function increases quickly with
   // the dimensions of the map.  The following figures are for a single
   // Intel Core2 quad-core processor at 3.4GHz running a single thread.
   //
   //   t = (56 + 5.3e-8 Ndim^4.2 + 4.9e-9 Ndim^4.2) seconds
   //         ^    ^               ^
   //         |    |               |
   //         |    |                \
   //         |    |                 \--- compute the homogeneous part
   //         |    |
   //         |     \--- compute the Greens function integral
   //         |
   //          \--- compute the Greens function (35x35 discrete kernel)
   //
   // Example: for a 1000 x 1000 pixel map, t = 229,000 seconds (64 hr)

#if REPORT_PROCESS_TIME
   struct timeval time_so_far;
   struct rusage resource_usage[9];
   getrusage(RUSAGE_SELF, &resource_usage[0]);
   printf("-- Entry to Map2D::PoissonSolve, starting the process clock.\n");
   printf("-- time to compute the Greens function for this grid: ");
   std::cout << std::flush;
#endif

   Int_t nxbins = rho->GetNbinsX();
   Int_t nybins = rho->GetNbinsY();
   Double_t xmin = rho->GetXaxis()->GetXmin();
   Double_t xmax = rho->GetXaxis()->GetXmax();
   Double_t ymin = rho->GetYaxis()->GetXmin();
   Double_t ymax = rho->GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   Double_t xrange = xmax-xmin-dx/2;
   Double_t yrange = ymax-ymin-dy/2;
   Map2D *G = (Map2D*)gROOT->FindObject("G");
   if (G == 0 || G->GetNbinsX() != 2*nxbins-1 ||
                 G->GetNbinsY() != 2*nybins-1 ||
                 G->GetXaxis()->GetXmin() != -xrange ||
                 G->GetXaxis()->GetXmax() != +xrange ||
                 G->GetYaxis()->GetXmin() != -yrange ||
                 G->GetYaxis()->GetXmax() != +yrange)
   {
      if (G)
         delete G;
      G = new Map2D(TH2D("G","2D Green\'s Function",
                         2*nxbins-1,-xrange,xrange,
                         2*nybins-1,-yrange,yrange));
      G->GreensFunction(0,0);
   }
   Int_t igx0 = nxbins;
   Int_t igy0 = nybins;

#if REPORT_PROCESS_TIME
   getrusage(RUSAGE_SELF, &resource_usage[1]);
   timersub(&resource_usage[1].ru_utime,
            &resource_usage[0].ru_utime,
            &time_so_far);
   printf("%f\n",time_so_far.tv_sec + time_so_far.tv_usec/1000000.);
   printf("-- time to do the Greens function integral for this grid: ");
   std::cout << std::flush;
#endif
   
   Map2D *work = new Map2D(*rho);
   work->SetName("work");
   work->Reset();
   for (Int_t ix0=1; ix0 <= nxbins; ++ix0) {
      for (Int_t iy0=1; iy0 <= nybins; ++iy0) {
         Double_t q = rho->GetBinContent(ix0,iy0) * dx*dy;
         if (q != 0) {
            for (Int_t ix=1; ix <= nxbins; ++ix) {
               for (Int_t iy=1; iy <= nybins; ++iy) {
                  Double_t sol = work->GetBinContent(ix,iy);
                  sol += q*G->GetBinContent(ix-ix0+igx0,iy-iy0+igy0);
                  work->SetBinContent(ix,iy,sol);
               }
            }
         }
      }
   }

#if REPORT_PROCESS_TIME
   getrusage(RUSAGE_SELF, &resource_usage[2]);
   timersub(&resource_usage[2].ru_utime,
            &resource_usage[1].ru_utime,
            &time_so_far);
   printf("%f\n",time_so_far.tv_sec + time_so_far.tv_usec/1000000.);
   printf("-- time to compute the homogeneous solution for this grid: ");
   std::cout << std::flush;
#endif

   Map2D boundarymap(*this);
   Add(work,-1);
   Mask(&boundarymap);
   Int_t result = LaplaceSolve(mask);
   Add(work,+1);

#if REPORT_PROCESS_TIME
   getrusage(RUSAGE_SELF, &resource_usage[3]);
   timersub(&resource_usage[3].ru_utime,
            &resource_usage[2].ru_utime,
            &time_so_far);
   printf("%f\n",time_so_far.tv_sec + time_so_far.tv_usec/1000000.);
   printf("-- Exit from Map2D::PoissonSolve.\n");
#endif

   return result;
}

Double_t Map2D::PoissonIter(const Map2D *rho, const Map2D *mask)
{
   // Iterates the Poisson equation in two dimensions with a source given by
   // input map rho, subject to the boundary conditions contained in this
   // object at entry.  If input argument "mask" points to an existing object,
   // that object is interpreted as a mask for the pixels in map *this that
   // contain boundary values, otherwise all non-zero pixels in map *this
   // at input are considered to specify boundary values.  The algorithm
   // will make use of as many (or few) boundary values as the user provides,
   // although the solution may not satisfy all of them exactly if the
   // problem is over-determined, i.e. too many values are provided, or they
   // are clustered in too compact a region of the map.  In any case, the
   // solution will attempt to match all of the provided data points as
   // closely as possible, subject to the constraints of Poisson's equation
   // and the resolution of the map.  All cells in the map *this are over
   // written with the solution except those specifying boundary conditions.
   //
   // Return value:
   // The largest correction that was applied to any cell in the map in the
   // present iteration, or zero if no changes were made.
   //
   // Algorithm:
   // The following formula is applied to all cells except those with rho=0
   // and the ones that specify boundary conditions.  Boundary condition 
   // cells are identified by having a non-zero value in the corresponding
   // cell of the mask map, or if mask=0, by having non-zero values in the
   // *this map at entry.
   //
   //           / (n)    (n)   \    2   / (n)    (n)   \   2            2  2
   //          | u   +  u       | dy + | u   +  u       | dx  - rho   dx dy
   //  (n+1)    \ i-1,j  i+1,j /        \ i,j-1  i,j+1 /           i,j
   // u     =   -------------------------------------------------------------
   //  i,j                        /  2      2 \                            .
   //                          2 | dx  +  dy   |
   //                             \           /                            .
   //
   // Timing:
 
#if REPORT_PROCESS_TIME
   struct timeval time_so_far;
   struct rusage resource_usage[9];
   getrusage(RUSAGE_SELF, &resource_usage[0]);
#endif

   Int_t nxbins = rho->GetNbinsX();
   Int_t nybins = rho->GetNbinsY();
   Double_t xmin = rho->GetXaxis()->GetXmin();
   Double_t xmax = rho->GetXaxis()->GetXmax();
   Double_t ymin = rho->GetYaxis()->GetXmin();
   Double_t ymax = rho->GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   Double_t dx2 = dx*dx;
   Double_t dy2 = dy*dy;
   Double_t dx2dy2 = dx2*dy2;

   Map2D delta(TH2D("delta","delta",nxbins,xmin,xmax,nybins,ymin,ymax));

   for (int ix=2; ix < nxbins; ++ix) {
      for (int iy=2; iy < nybins; ++iy) {
         double rho_i_j = (rho == 0)? 0 : rho->GetBinContent(ix,iy);
         if (rho_i_j == 0) {
            continue;
         }
         double u_i_j = GetBinContent(ix,iy);
         double u_im1_j = GetBinContent(ix-1,iy);
         double u_ip1_j = GetBinContent(ix+1,iy);
         double u_i_jm1 = GetBinContent(ix,iy-1);
         double u_i_jp1 = GetBinContent(ix,iy+1);
         if ((mask != 0 && mask->GetBinContent(ix,iy) == 0) || 
             (mask == 0 && GetBinContent(ix,iy) == 0))
         {
            double new_u_i_j = (u_im1_j + u_ip1_j) * dy2
                             + (u_i_jm1 + u_i_jp1) * dx2
                             - rho_i_j * dx2dy2;
            new_u_i_j /= 2 * (dx2 + dy2);
            if (u_im1_j * u_ip1_j * u_i_jm1 * u_i_jp1 != 0) {
               delta.SetBinContent(ix,iy, new_u_i_j - u_i_j);
            }
         }
      }
   }

   Add(&delta);

#if REPORT_PROCESS_TIME
   getrusage(RUSAGE_SELF, &resource_usage[1]);
   timersub(&resource_usage[1].ru_utime,
            &resource_usage[0].ru_utime,
            &time_so_far);
   printf("Map2D::PoissonIter used %f cpu seconds\n",
          time_so_far.tv_sec + time_so_far.tv_usec/1000000.);
#endif

   double dmax = fabs(delta.GetMaximum());
   double dmin = fabs(delta.GetMinimum());
   return (dmax > dmin)? dmax : dmin;
}

Bool_t Map2D::GreensFunction(Double_t x0, Double_t y0)
{
   //  The Green's function for the 2D Poisson equation can be written most
   //  compactly as follows,
   //
   //                             1        |      2        2 |
   //             G(x,y;x ,y ) = ----  log |(x-x ) + (y-y )  |
   //                    0  0    4 pi      |    0        0   |
   //
   // where x0,y0 is the location of the source.  In two dimensions it is
   // not possible to set the boundary conditions for G->0 at x0,y0->infinity
   // so instead one choses that G=0 at some finite radius R.  The expression
   // above has been written with R=1, but adding a constant term can be used
   // to shift R to an arbitrary radius, except for the singular values 0 and
   // infinity.  The result is loaded into this map, overwriting any contents
   // that were present at input, but leaving the dimensions and bounds of
   // the original map unchanged.  The point x0,y0 is assumed to lie somewhere
   // within the bounds of the map.
   //
   // This is the correct expression for the continuum case, but it diverges
   // as x,y -> x0,y0, and has large discretization errors for pixels nearby.
   // For this reason, I take a NxN pixel square region centered around x0,y0
   // and replace the computed value of G for these pixels with the solution
   // to the discrete NxN matrix Poisson equation on a square grid, subject
   // to the boundary condition that the cells around the outer perimeter of
   // the NxN pixel block match onto the above continuum formula.  By chosing
   // N to be sufficiently large, the residual discretization errors are kept
   // to a negligible level.

   Int_t nxbins = GetNbinsX();
   Int_t nybins = GetNbinsY();
   Double_t xmin = GetXaxis()->GetXmin();
   Double_t xmax = GetXaxis()->GetXmax();
   Double_t ymin = GetYaxis()->GetXmin();
   Double_t ymax = GetYaxis()->GetXmax();
   Double_t dx = (xmax-xmin)/nxbins;
   Double_t dy = (ymax-ymin)/nybins;
   Int_t ix0 = int((x0-xmin)/dx)+1;
   Int_t iy0 = int((y0-ymin)/dy)+1;
   Double_t fourpi=8*acos(0);
   for (Int_t ix=1; ix <= nxbins; ++ix) {
      Double_t x = xmin+(ix-0.5)*dx-x0;
      for (Int_t iy=1; iy <= nybins; ++iy) {
         Double_t y = ymin+(iy-0.5)*dy-y0;
         Double_t r2 = x*x+y*y+1e-50;
         SetBinContent(ix,iy,log(r2)/fourpi);
      }
   }

   // Cut out a square region around the source and solve the discrete problem

   Int_t Ndim = 35;
   Int_t ix1 = ix0-Ndim/2;
   Int_t iy1 = iy0-Ndim/2;
   if (ix0 < 3 || ix0 > nxbins-4 ||
       iy0 < 3 || ix0 > nxbins-4) {
      std::cerr << "Map2D::GreensFunction error - computing the Green\'s"
                << " function with x0,y0 too close to the edge of the map,"
                << " cannot continue."
                << std::endl;
      return false;
   }
   else if (ix1 < 0 || ix1+Ndim >= nxbins ||
            iy1 < 0 || iy1+Ndim >= nybins) {
      std::cerr << "Map2D::GreensFunction warning - computing the Green\'s"
                << " function with x0,y0 close to the edge of the map,"
                << " loss of precision will result."
                << std::endl;
      Int_t margin=0;
      margin = (-ix1 > margin)? -ix1 : margin;
      margin = (-iy1 > margin)? -iy1 : margin;
      margin = (ix1+Ndim-nxbins >= margin)? ix1+Ndim-nxbins+1 : margin;
      margin = (iy1+Ndim-nybins >= margin)? iy1+Ndim-nybins+1 : margin;
      Ndim -= margin;
      ix1 = ix0-Ndim/2;
      iy1 = iy0-Ndim/2;
   }

   // Set up the discretized Poisson equation in the form
   //           A x = b
   // where A is a square matrix and x,b are column vectors of the same
   // dimension as A.  There is one row in this equation for each pixel
   // in a NxN pixel block surrounding the one pixel at ix0,iy0 where the
   // source is located.  The (N-2)*(N-2) rows in A corresponding to the
   // interior pixels of the block contain the discrete approximation to
   // the divergence operator.  For the 4N boundary pixels, A contains
   // just a 1 on the diagonal to enforce the constraints that the boundary
   // pixels should match the values computed above using the continuum
   // formula.  The b vector contains zero for all of the interior pixels
   // except the one at ix0,iy0 which is set to 1/(dx*dy), the discrete
   // approximation to the Dirac delta function in two dimensions.  For
   // the boundary pixels, b is set to the values computed using the
   // continuum formula.

   TMatrixD A(Ndim*Ndim,Ndim*Ndim);
   TVectorD b(Ndim*Ndim);
   A.Zero();
   Int_t eq=0;
   for (Int_t ix=0; ix < Ndim; ++ix) {
      for (Int_t iy=0; iy < Ndim; ++iy,++eq) {
         Int_t ci = iy*Ndim+ix;
         if (ix == 0 || ix == Ndim-1 || iy == 0 || iy == Ndim-1) {
            A(eq,ci) = 1;
            b(eq) = GetBinContent(ix+ix1,iy+iy1);
         }
         else {
            A(eq,ci) = -2/(dx*dx)-2/(dy*dy);
            A(eq,ci-Ndim) = 1/(dy*dy);
            A(eq,ci+Ndim) = 1/(dy*dy);
            A(eq,ci-1) = 1/(dx*dx);
            A(eq,ci+1) = 1/(dx*dx);
            b(eq) = (ix+ix1 == ix0 && iy+iy1 == iy0)? 1/(dx*dy) : 0;
         }
      }
   }

   // Use singular value decomposition to solve

   TDecompSVD svd(A);
   if (!svd.Decompose()) {
      std::cerr << "Map2D::GreensFunction error - TDecompSVD method"
                << " Decompose failed, unable to continue."
                << std::endl;
      return false;
   }
   svd.Solve(b);
   for (Int_t ix=0; ix < Ndim; ++ix) {
      for (Int_t iy=0; iy < Ndim; ++iy) {
         Int_t ci = iy*Ndim+ix;
         SetBinContent(ix+ix1,iy+iy1,b(ci));
      }
   }
   return true;
}

Int_t Map2D::LaplaceSolve(const Map2D *mask)
{
   // Solves the Laplace equation in two dimensions, subject to the boundary
   // conditions contained in this object at entry.  If input argument "mask"
   // points to an existing object, that object is interpreted as a mask for
   // the pixels in map *this that contain boundary values, otherwise all
   // non-zero pixels in map *this at input are considered to specify
   // boundary values.  The algorithm will make use of as many (or few)
   // boundary values as the user provides, although the solution may not
   // satisfy all of them exactly if the problem is over-determined, i.e.
   // too many values are provided, or they are clustered in too compact a
   // region of the map.  In any case, the solution will attempt to match all
   // of the provided data points as closely as possible, subject to the
   // constraints of the Laplace equation and the resolution of the map.
   // All cells in the map *this are over written with the solution.
   //
   // Algorithm:
   // The problem is solved in the basis of solutions to Laplace's equation
   // in polar coordinates.  Here f_n is a stackable column vector of two
   // basis functions
   //
   //                             /                              .
   //                         m  |  cos(m phi)
   //            f (x,y) =   r  < 
   //             m              |  sin(m phi)
   //                             \                              .
   //
   // There is a second set of solutions that are singular at the origin,
   // but these are excluded because the interior region of the map is
   // assumed to include the origin, taken to lie at the geometric center
   // of the map.  Note that for n=0 the second element of f_n is identically
   // zero.  To form a column vector of basis functions F_i, imagine a stack
   // of f_m column vectors starting with m=0 at the top, and increasing
   // downward until the number N of elements in F equals at least M, the
   // number the pixels (x_j,y_j), j=0..M that are provided as boundary
   // values in the input map.  This results in a large MxN matrix A_ji
   // such that
   //                   A     = F ( x , y )
   //                    j,i     i   j   j
   //
   // Solving Laplace's equation now amounts to solving the linear equation
   //
   //                   A    C  = z
   //                    j,i  i    j
   //
   // for unknown coefficients C_i in terms of initial values z_j for pixel j.
   // In general A is not a square matrix, and has many singular values.  I
   // solve this equation using singular value decomposition, and identify
   // the rank of the matrix A, which is passed as the return value from this
   // method.  The results of the singular value decomposition are used to
   // filter the given values z_j into the unknown coefficients C_i, which 
   // are then used to fill the map with the optimal solution to the boundary
   // value problem.
   //
   // The user should pay careful attention to this value, as it may
   // indicate that the problem is ill-posed.  In general, one expects that
   // the rank will be close to the number of boundary values provided.  It
   // will never be greater.  If the rank is significantly less than the
   // number of boundary values provided, it may indicate that the boundary
   // values over-determine the solution, or it may simply mean that fewer
   // basis functions are needed to describe the solution than the number
   // of boundary values would naively suggest.

   class basis_function {
    public:

      basis_function(const Map2D *map)
      {
         fNxbins = map->GetNbinsX();
         fNybins = map->GetNbinsY();
         fXlow = map->GetXaxis()->GetXmin();
         fXhigh = map->GetXaxis()->GetXmax();
         fYlow = map->GetYaxis()->GetXmin();
         fYhigh = map->GetYaxis()->GetXmax();
         fXwidth = fXhigh-fXlow;
         fYwidth = fYhigh-fYlow;
         fXcenter = (fXhigh+fXlow)/2;
         fYcenter = (fYhigh+fYlow)/2;
         fXdelta = (fXhigh-fXlow)/fNxbins;
         fYdelta = (fYhigh-fYlow)/fNybins;
         fRscale2 = (fXwidth*fXwidth + fYwidth*fYwidth)/4;
      }

      Double_t F(Int_t i, Double_t x, Double_t y)
      {
         x -= fXcenter;
         y -= fYcenter;
         Double_t r2 = (x*x+y*y)/fRscale2;
         Double_t phi = atan2(y,x);
         Int_t m = (i+1)/2;
         switch (i-m*2) {
          case 0:
            return pow(r2,m/2.) * cos(m*phi);
          case -1:
            return pow(r2,m/2.) * sin(m*phi);
          default:
            std::cerr << "Map2D::LaplaceSolve::basis_function error - "
                      << "method F called with negative index i = "
                      << i << std::endl;
            return 0;
         }
      }

      Double_t F(Int_t i, Int_t ix, Int_t iy)
      {
         return F(i,(ix-0.5)*fXdelta+fXlow,(iy-0.5)*fYdelta+fYlow);
      }

    private:
      Int_t fNxbins;
      Int_t fNybins;
      Double_t fXlow;
      Double_t fXhigh;
      Double_t fYlow;
      Double_t fYhigh;
      Double_t fXwidth;
      Double_t fYwidth;
      Double_t fXcenter;
      Double_t fYcenter;
      Double_t fXdelta;
      Double_t fYdelta;
      Double_t fRscale2;
   };

   Int_t nxbins = GetNbinsX();
   Int_t nybins = GetNbinsY();

   // Make a list of input boundary values

   std::vector<Double_t> bX,bY,bValue;
   std::vector<Int_t> iX,iY;
   for (Int_t ix=1; ix <= nxbins; ++ix) {
      for (Int_t iy=1; iy <= nybins; ++iy) {
         if ((mask != 0 && mask->GetBinContent(ix,iy) != 0) ||
             (mask == 0 && GetBinContent(ix,iy) != 0))
         {
            iX.push_back(ix);
            iY.push_back(iy);
            bX.push_back(GetXaxis()->GetBinCenter(ix));
            bY.push_back(GetYaxis()->GetBinCenter(iy));
            bValue.push_back(GetBinContent(ix,iy));
         }
      }
   }

   // Fill the A matrix with basis functions evaluated at bValue points

   basis_function f(this);
   Int_t Mvalues = bValue.size();
   Int_t Nbases = Mvalues;
   Mvalues = (Mvalues < Nbases)? Nbases : Mvalues;
   TMatrixD A(Mvalues,Nbases);
   TVectorD b(Mvalues);
   for (UInt_t m=0; m < bValue.size(); ++m) {
      for (Int_t n=0; n < Nbases; ++n) {
         A(m,n) = f.F(n,bX[m],bY[m]);
      }
      b(m) = bValue[m];
   }

   // Add null rows and bValue points to bring Mvalues up to at least Nbases

   for (Int_t m=bValue.size(); m < Mvalues; ++m) {
      for (Int_t n=0; n < Nbases; ++n) {
         A(m,n) = 0;
      }
      b(m) = 0;
   }

   // Compute the SVD of A and compute its rank

   TDecompSVD svd(A);
   if (!svd.Decompose()) {
      std::cerr << "Map2D::LaplaceSolve error - TDecompSVD method"
                << " Decompose failed, unable to continue."
                << std::endl;
      return 0;
   }
   svd.Solve(b);
   TVectorD sig = svd.GetSig();
   Int_t rank;
   for (rank=0; rank < sig.GetNrows(); ++rank) {
      if (sig(rank) < 1e-300) {
         break;
      }
   }

   // Overwrite the map with this solution

   for (Int_t ix=1; ix <= nxbins; ++ix) {
      for (Int_t iy=1; iy <= nybins; ++iy) {
         Double_t val=0;
         for (Int_t n=0; n < Nbases; ++n) {
            val += b(n)*f.F(n,ix,iy);
         }
         SetBinContent(ix,iy,val);
      }
   }
#if LAPLACE_PRINT_BOUNDARY_MATCHING
   for (unsigned int m=0; m < bValue.size(); ++m) {
      double val=0;
      for (int n=0; n < Nbases; ++n) {
         val += b(n)*f.F(n,bX[m],bY[m]);
      }
      printf("point %d: input %f output %f\n",m,bValue[m],val);
   }
#endif
   return rank;
}

TH1D &Map2D::Profile(const char *name, const char *title,
                     Int_t nbins, Double_t xlow, Double_t xhigh)
{
   // Scans over the map and forms a histogram of the cell heights,
   // omitting the cells that contain zero.

   if (xlow == 999) {
      xlow = 1e30;
      for (Int_t ix=1; ix <= GetNbinsX(); ++ix) {
         for (Int_t iy=1; iy <= GetNbinsY(); ++iy) {
            double cont = GetBinContent(ix,iy);
            if (cont == 0) {
               continue;
            }
            xlow = (cont < xlow)? cont : xlow;
         }
      }
   }
   if (xhigh == 999) {
      xhigh = -1e30;
      for (Int_t ix=1; ix <= GetNbinsX(); ++ix) {
         for (Int_t iy=1; iy <= GetNbinsY(); ++iy) {
            double cont = GetBinContent(ix,iy);
            if (cont == 0) {
               continue;
            }
            xhigh = (cont > xhigh)? cont : xhigh;
         }
      }
   }

   TH1D *prof = new TH1D(name,title,nbins,xlow,xhigh);
   for (Int_t ix=1; ix <= GetNbinsX(); ++ix) {
      for (Int_t iy=1; iy <= GetNbinsY(); ++iy) {
         double cont = GetBinContent(ix,iy);
         if (cont) {
            prof->Fill(cont);
         }
      }
   }
   return *prof;
}

Map2D *Map2D::copy(const char *name) const
{
   // Protected method to do a "safe" clone operation on a Map2D object
   // where "safe" means to test if an object with the desired name
   // already exists and if so, delete it first, then create the clone.
   // This method MUST NOT be called with name == this->GetName().

   TObject *old = gROOT->FindObject(name);
   if (old) {
      std::cerr << "Warning in Map2D::copy - overwriting existing map "
                << "named " << name << std::endl;
      delete old;
   }
   Map2D *copy = new Map2D(*this);
   copy->SetName(name);
   return copy;
}

void Map2D::clone(const Map2D *src)
{
   // Copy the contents of *src to *this, but retain the original
   // name and title of *this.

   std::string name(GetName());
   std::string title(GetTitle());
   *this = *src;
   SetName(name.c_str());
   SetTitle(title.c_str());
}
