#!/usr/bin/env python
"""
Simulate irradiation → damage map → spectral scans (Kellie-style).

Produces:
  1. Dose map (analogous to Kellie Fig. 7)
  2. Edge width vs x (Kellie Fig. 10)
  3. Symmetric spectra at ±x with zoom (Kellie Fig. 11)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gonio_sim.envs.goniometer_env import GoniometerEnv, EnvConfig
from gonio_sim.physics.root_io import init_root
from gonio_sim.physics.damage_model import DiamondDamageModel


# ---------------------------------------------
# Spectrum helpers
# ---------------------------------------------

def get_coherent_spectrum(env):
    """Compute spectrum at current base_args and apply damage."""
    ROOT = init_root(env.base_args)

    h = ROOT.cobrems_intensity(
        env.base_args["radname"], env.base_args["iradview"],
        env.base_args["ebeam"], env.base_args["ibeam"],
        env.base_args["xyresol"], env.base_args["thetah"], env.base_args["thetav"],
        env.base_args["xoffset"], env.base_args["yoffset"], env.base_args["phideg"],
        env.base_args["xsigma"], env.base_args["ysigma"], env.base_args["xycorr"],
        env.base_args["peresol"], env.base_args["penergy0"], env.base_args["penergy1"], 0
    )[0]

    n = h.GetNbinsX()
    E_GeV = np.array([h.GetBinCenter(i+1) for i in range(n)])
    I = np.array([h.GetBinContent(i+1) for i in range(n)])

    # damage applied here (energy domain)
    dm = env.damage
    dose = env.dose
    I = dm.apply(1000.0 * E_GeV, I, dose)

    return 1000.0 * E_GeV, I  # return MeV


def edge_width(E_MeV, I):
    """Compute coherent-edge width ~ Kellie Fig. 9."""
    imax = np.argmax(I)

    lo = max(0, imax - 40)
    hi = min(len(E_MeV), imax + 80)

    E = E_MeV[lo:hi]
    y = I[lo:hi]

    ymax, ymin = y.max(), y.min()

    def crossing(level):
        diff = y - level
        idxs = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]
        if not len(idxs):
            return None
        i = idxs[-1]
        x0, x1 = E[i], E[i+1]
        y0, y1 = y[i], y[i+1]
        return x0 if y0 == y1 else x0 + (level - y0)*(x1 - x0)/(y1 - y0)

    x_top = crossing(ymax)
    x_bot = crossing(ymin)

    return abs(x_top - x_bot) if (x_top and x_bot) else np.nan


def plot_full_dose_map(env, title="Full Dose Map (arbitrary units)"):
    """
    Plot the entire 70×70 mm dose map using arbitrary units.
    """
    plt.figure(figsize=(7,6))
    plt.imshow(
        env.dose_map.T,
        origin="lower",
        extent=[env.grid_x[0], env.grid_x[-1],
                env.grid_y[0], env.grid_y[-1]],
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Dose (arb. units)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    plt.tight_layout()

def plot_zoomed_dose_map(env, center_x_mm=1.0, center_y_mm=0.0,
                         width_mm=4.0, height_mm=4.0,
                         title="Zoomed Dose Map (arb. units)"):
    """
    Plot a zoomed-in region of the dose map around a chosen (x,y) center.
    Units of dose are arbitrary (not physical).
    """

    xmin = center_x_mm - width_mm/2
    xmax = center_x_mm + width_mm/2
    ymin = center_y_mm - height_mm/2
    ymax = center_y_mm + height_mm/2

    # mask to extract zoom window
    xmask = (env.grid_x >= xmin) & (env.grid_x <= xmax)
    ymask = (env.grid_y >= ymin) & (env.grid_y <= ymax)

    zoom_map = env.dose_map[np.ix_(xmask, ymask)]

    plt.figure(figsize=(6,5))
    plt.imshow(
        zoom_map.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Dose (arb. units)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    plt.tight_layout()

    

# ---------------------------------------------
# Main simulation driver
# ---------------------------------------------

def main():
    env = GoniometerEnv(EnvConfig())

    # Initialize geometry (sets thetah/thetav)
    env.step(0,0)

    # Tune damage model (demo values)
    env.damage.alpha  = 0.003   # MeV per dose
    env.damage.beta1  = 0.001   # MeV per dose
    env.damage.D0     = 200.0
    env.damage.sigma0 = 2.0

    # dose per irradiation step
    env.dose_per_step = 1.0

    # Center the damage region at +1 mm to break symmetry
    x0_damage = 1.0
    y0_damage = 0.0
    spot_sigma = 0.3  # mm

    # 1. IRRADIATION (FAST)
    n_steps = 100000
    print("Irradiating the diamond...")

    for _ in tqdm(range(n_steps), desc="Irradiation"):
        x = np.random.normal(x0_damage, spot_sigma)
        y = np.random.normal(y0_damage, spot_sigma)
        env.deposit_dose_gaussian(x, y, sigma=0.3)

    # ---------------------------------------
    # SHOW FULL 70×70 mm DOSE MAP
    # ---------------------------------------
    plot_full_dose_map(env, title="Full Radiation Dose Map (e⁻/mm²)")

    # ---------------------------------------
    # SHOW ZOOMED-IN WINDOW AROUND +1.0 cm
    # ---------------------------------------
    plot_zoomed_dose_map(
        env,
        center_x_mm=1.0,   # +1.0 cm
        center_y_mm=0.0,
        width_mm=4.0,       # 4×4 mm window (adjustable)
        height_mm=4.0,
        title="Zoomed Dose Map at +1.0 cm"
    )

    # Freeze dose during scans
    env.dose_per_step = 0.0

    # 2. EDGE WIDTH VS X
    xs = np.linspace(-2.0, 2.0, 21)
    widths = []

    print("Scanning coherent-edge width vs x...")

    for x in tqdm(xs, desc="Edge width scan"):
        env.base_args["xoffset"] = x
        env.base_args["yoffset"] = 0.0
        env.step(0,0)           # this updates env.dose from dose_map

        E, I = get_coherent_spectrum(env)
        widths.append(edge_width(E, I))

    plt.figure(figsize=(7,5))
    plt.plot(xs, widths, marker='o')
    plt.xlabel("x (mm)")
    plt.ylabel("Edge width (MeV)")
    plt.title("Coherent-Edge Width vs x (Kellie Fig. 10)")
    plt.grid(True)
    plt.tight_layout()

    # 3. SYMMETRIC SPECTRA
    displacements = [0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()

    print("Computing symmetric spectra at ±x...")

    for ax, dx in tqdm(list(zip(axes, displacements)), desc="Symmetric spectra"):

        # left point
        env.base_args["xoffset"] = -dx
        env.base_args["yoffset"] = 0.0
        env.step(0,0)
        E_L, I_L = get_coherent_spectrum(env)
        I_L /= I_L.max()

        # right point
        env.base_args["xoffset"] = dx
        env.step(0,0)
        E_R, I_R = get_coherent_spectrum(env)
        I_R /= I_R.max()

        # zoom window
        Emin, Emax = 7000, 9000
        mask = (E_L > Emin) & (E_L < Emax)
        diff = I_L - I_R

        ax.plot(E_L[mask], I_L[mask], label=f"x={-dx} mm")
        ax.plot(E_R[mask], I_R[mask], '--', label=f"x={dx} mm")
        #ax.plot(E_L[mask], 10*diff[mask], 'k:', label="10× (left - right)")

        ax.set_title(f"Symmetric spectra at ±{dx} mm")
        ax.set_xlabel("Photon Energy (MeV)")
        ax.set_ylabel("Relative Intensity")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
