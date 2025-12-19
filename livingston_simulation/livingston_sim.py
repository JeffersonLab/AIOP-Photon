import numpy as np

def orientation_to_phi_deg(orientation):

    orientation = orientation.strip().upper()

    mapping = {
        "PARA 0/90":   0.0,
        "PERP 0/90":  90.0,
        "PARA 45/135": 135.0,
        "PERP 45/135": 45.0,
    }

    if orientation not in mapping:
        raise ValueError(
            f"Unknown orientation '{orientation}'. "
            f"Valid options: {list(mapping.keys())}"
        )

    return mapping[orientation]



# ============================================================
# USER-DEFINED PHYSICS HOOKS (FILL THESE IN)
# ============================================================

def delta_c_from_pitch_yaw(
    pitch_change_deg,
    yaw_change_deg,
    orientation: CrystalOrientation,
    beam_pitch_deg=0.0,
    beam_yaw_deg=0.0
):

        raise NotImplementedError

    else:
        raise ValueError(f"Unknown orientation: {orientation}")


def delta_c_to_peak_shift(delta_c_rad, E0, Ei):
    g = 2
    k = 26.5601
    deltaE = (delta_c * (E0 - Ei)**2 ) / (k/g + delta_c * (E0 - E) )
    return deltaE



class GoniometerState:
    """
    Tracks goniometer angles, beam alignment,
    and crystal orientation (via phi).
    """

    _ORIENTATION_TO_PHI = {
        "PARA 0/90":    0.0,
        "PERP 0/90":   90.0,
        "PARA 45/135": 135.0,
        "PERP 45/135": 45.0,
    }

    def __init__(
        self,
        pitch_deg=0.0,
        yaw_deg=0.0,
        beam_pitch_deg=0.0,
        beam_yaw_deg=0.0,
        orientation="PARA 0/90"
    ):
        self.pitch_deg = pitch_deg
        self.yaw_deg = yaw_deg
        self.beam_pitch_deg = beam_pitch_deg
        self.beam_yaw_deg = beam_yaw_deg

        self.set_orientation(orientation)

    def set_orientation(self, orientation):
        """
        Set crystal orientation using a human-readable label.
        """

        key = orientation.strip().upper()

        if key not in self._ORIENTATION_TO_PHI:
            raise ValueError(
                f"Unknown orientation '{orientation}'. "
                f"Valid options: {list(self._ORIENTATION_TO_PHI.keys())}"
            )

        self.orientation = key
        self.phi_deg = self._ORIENTATION_TO_PHI[key]

    def move(self, dpitch_deg, dyaw_deg):
        self.pitch_deg += dpitch_deg
        self.yaw_deg += dyaw_deg



class DiamondDose:
    """
    Tracks accumulated diamond dose.
    """

    def __init__(self, dose=0.0):
        self.dose = dose

    def add(self, delta_dose):
        self.dose += delta_dose

    def reset(self):
        self.dose = 0.0


class CoherentPeakTracker:

    def __init__(
        self,
        base_peak_position,
        dose_slope
    ):
        self.base_peak_position = base_peak_position
        self.dose_slope = dose_slope

    def peak_position(self, delta_c_rad, dose):
        peak_shift_c = delta_c_to_peak_shift(delta_c_rad)
        peak_shift_dose = self.dose_slope * dose

        return (
            self.base_peak_position
            + peak_shift_c
            + peak_shift_dose
        )


# ============================================================
# HIGH-LEVEL SIMULATOR
# ============================================================

class CoherentBremsstrahlungSimulator:
    """
    Combines:
    - goniometer + beam state
    - crystal orientation
    - dose tracking
    - coherent peak model
    """

    def __init__(
        self,
        base_peak_position,
        dose_slope,
        orientation=CrystalOrientation.PARA_0_90
    ):
        self.state = GoniometerState(orientation=orientation)
        self.dose = DiamondDose()
        self.peak = CoherentPeakTracker(
            base_peak_position=base_peak_position,
            dose_slope=dose_slope
        )

    def step(
        self,
        dpitch_deg=0.0,
        dyaw_deg=0.0,
        delta_dose=0.0
    ):
        """
        Apply a goniometer move and dose increment.
        Returns updated peak position.
        """

        self.state.move(dpitch_deg, dyaw_deg)
        self.dose.add(delta_dose)

        delta_c = delta_c_from_pitch_yaw(
            pitch_change_deg=self.state.pitch_deg,
            yaw_change_deg=self.state.yaw_deg,
            orientation=self.state.orientation,
            beam_pitch_deg=self.state.beam_pitch_deg,
            beam_yaw_deg=self.state.beam_yaw_deg
        )

        return self.peak.peak_position(
            delta_c_rad=delta_c,
            dose=self.dose.dose
        )

# ============================================================
# MAIN SIMULATION
# ============================================================

def main():
    sim = CoherentBremsstrahlungSimulator(
        base_peak_position=100.0,   # MeV
        dose_slope=0.05,            # MeV per dose unit
        orientation=CrystalOrientation.PARA_45_135
    )

    # Beam misalignment
    sim.state.beam_pitch_deg = 0.03
    sim.state.beam_yaw_deg = -0.02

    print("Step | Pitch(deg) | Yaw(deg) | Dose | Delta c | Peak")
    print("-" * 60)

    # Simple scan
    for i in range(10):
        delta_c, peak = sim.step(
            dpitch_deg=0.01,
            dyaw_deg=-0.005,
            delta_dose=0.2
        )

        print(
            f"{i:4d} | "
            f"{sim.state.pitch_deg:10.4f} | "
            f"{sim.state.yaw_deg:8.4f} | "
            f"{sim.dose.dose:4.1f} | "
            f"{delta_c:7.4f} | "
            f"{peak:7.3f}"
        )


if __name__ == "__main__":
    main()
