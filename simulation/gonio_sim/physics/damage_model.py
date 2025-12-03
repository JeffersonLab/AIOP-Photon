# damage_model.py
import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift

class DiamondDamageModel:
    def __init__(self, alpha=0.0, beta1=0.0, beta2=0.0, D0=np.inf, sigma0=0.0):
        """
        alpha  : broadening growth per dose  [MeV / (e-/cm^2)]
        beta1  : linear peak shift per dose  [MeV / (e-/cm^2)]
        beta2  : quadratic term (optional)   [MeV / (e-/cm^2)^2]
        D0     : amplitude e-fold dose       [e-/cm^2]
        sigma0 : baseline energy blur (MeV) from undamaged mosaic/spread
        """
        self.alpha  = alpha
        self.beta1  = beta1
        self.beta2  = beta2
        self.D0     = D0
        self.sigma0 = sigma0

    def _gaussian_kernel(self, E, sigma):
        # normalized Gaussian in energy domain
        return np.exp(-0.5 * (E / (sigma + 1e-30))**2) / (np.sqrt(2*np.pi) * (sigma + 1e-30))

    def apply(self, Egamma, I0, dose):
        Egamma = np.asarray(Egamma)
        I0     = np.asarray(I0)

        # 1) amplitude attenuation
        A = np.exp(-dose / (self.D0 if np.isfinite(self.D0) else 1e300))
        I = A * I0.copy()

        # 2) shift the energy axis by ΔE(dose)
        dE = self.beta1 * dose + self.beta2 * dose * dose
        # shift by interpolation (keeps array length; extrapolate as 0)
        Eshift = Egamma - dE
        I = np.interp(Eshift, Egamma, I, left=0.0, right=0.0)

        # 3) Gaussian broadening via convolution
        sigma = np.sqrt(self.sigma0**2 + (self.alpha * dose)**2)
        if sigma > 0:
            # Use FFT-based convolution on a uniform grid
            dEgrid = np.mean(np.diff(Egamma))
            if not np.allclose(np.diff(Egamma), dEgrid, rtol=1e-6, atol=1e-9):
                # fallback to direct convolution if non-uniform grid
                # build kernel on relative energy
                ker_E = Egamma - Egamma[len(Egamma)//2]
                K = self._gaussian_kernel(ker_E, sigma)
                K /= (K.sum() * (Egamma[1]-Egamma[0]))
                I = np.convolve(I, K, mode='same')
            else:
                # uniform grid: FFT convolution
                n = len(Egamma)
                # build kernel centered at zero
                x = (np.arange(n) - n//2) * dEgrid
                K = self._gaussian_kernel(x, sigma)
                K = fftshift(K)
                Ik = ifft(fft(I) * fft(K)).real
                I = Ik

        return I
