#ifndef Map2D_h
#define Map2D_h

#include <TH1D.h>
#include <TH2D.h>

#include <limits>
#include <cmath>

class Map2D : public TH2D {
 public:
   Map2D();
   Map2D(const Map2D &m);
   Map2D(const TH2D &h);
   Map2D(const Map2D *m) : Map2D(*m) {}
   Map2D(const TH2D *h) : Map2D(*h) {}
   virtual ~Map2D();

   Map2D &operator=(const Map2D &src);

   using TH2D::Add;
   virtual Map2D &Add(const Map2D *map, Double_t c=1);
   virtual Map2D &Add(const Map2D *map1, const Map2D *map2,
                      Double_t c1=1, Double_t c2=1);
   using TH2D::Fill;
   virtual Map2D &Fill(const Map2D *map);
   virtual Map2D &Flood(Double_t value, 
                        Double_t xlow = -HUGE_VAL, Double_t ylow = -HUGE_VAL,
                        Double_t xhigh = HUGE_VAL, Double_t yhigh = HUGE_VAL);
   virtual Map2D &FillVoids();
   virtual Map2D &Smear(double xsigma, double ysigma);
   virtual Map2D &Shift(Double_t dx=0, Double_t dy=0, Double_t dz=0);
   virtual Map2D &Center();
   virtual Map2D &Level();
   virtual Map2D &Normalize();
   virtual Map2D &Rotate(Double_t phi=0, Double_t theta=0,
                         Double_t psi=0, Int_t degrees=0);
   virtual Map2D &Tilt(Double_t dzdx, Double_t dzdy,
                       Double_t x0=DBL_MAX, Double_t y0=DBL_MAX);
   virtual Map2D &Rescale(Double_t sx=1, Double_t sy=1, Double_t sz=1);
   virtual Map2D &Mask(const Map2D *map, Int_t neg=0, Double_t value=0);
   virtual Map2D &Resize(Double_t xlow, Double_t ylow,
                         Double_t xhigh, Double_t yhigh, Int_t pixels=0) const;

   virtual Int_t ZeroSuppress(Double_t threshold, Double_t zero=0);
   virtual Int_t Despeckle(Int_t maxsize=0);
   virtual Int_t Despike(Double_t maxheight, Int_t regionsize=3);
   virtual Double_t Correlation(const Map2D *map, Double_t contrast=1) const;
   virtual Double_t TotalCurl(const Map2D *Vx, const Map2D *Vy,
                              Map2D **silhouette=0);
   virtual Map2D &GetSilhouette();
   virtual Double_t Curl(const Map2D *Vx, const Map2D *Vy);
   virtual Map2D &Uncurl(const Map2D *Vx, const Map2D *Vy, Int_t comp);
   virtual Double_t Divergence(const Map2D *Vx, const Map2D *Vy);
   virtual Int_t Gradient(Map2D *Vx, Map2D *Vy) const;
   virtual Bool_t GreensFunction(Double_t x0, Double_t y0);
   virtual Double_t PoissonIter(const Map2D *rho=0, const Map2D *mask=0);
   virtual Int_t PoissonSolve(const Map2D *rho, const Map2D *mask=0);
   virtual Int_t LaplaceSolve(const Map2D *mask=0);

   virtual TH1D &Profile(const char *name="prof", const char *title="",
                Int_t nbins=100, Double_t xlow=999., Double_t xhigh=999.);

   static const Double_t zunit2xy;

   ClassDef(Map2D,0);

 protected:
   Map2D *rotateX(Double_t thetarad, Double_t y0, Double_t z0) const;
   Map2D *rotateY(Double_t thetarad, Double_t x0, Double_t z0) const;
   Map2D *rotateZ(Double_t phirad, Double_t x0, Double_t y0) const;
   Map2D *shift(Double_t dx, Double_t dy, Double_t dz) const;
   Map2D *rescale(Double_t sx, Double_t sy, Double_t sz,
                  Double_t x0, Double_t y0, Double_t z0) const;
   Map2D *smear(double xsigma, double ysigma);
   Map2D *copy(const char *name) const;
   void clone(const Map2D *src);
};

#endif
