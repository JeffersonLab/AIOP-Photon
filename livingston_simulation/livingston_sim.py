import numpy as np

# Canonical orientation labels and helpers shared across notebooks and the RL env
ORIENTATION_TO_PHI = {
    "AMORPHOUS":   0.0,
    "PARA 0/90":    0.0,
    "PERP 0/90":   90.0,
    "PARA 45/135": 135.0,
    "PERP 45/135": 45.0,
}

# Accept legacy labels used in notebooks (e.g., "0/90 PERP") and map to the canonical ones above
ORIENTATION_ALIASES = {
    "AMORPHOUS": "AMORPHOUS",
    "0/90 PERP": "PERP 0/90",
    "0/90 PARA": "PARA 0/90",
    "45/135 PERP": "PERP 45/135",
    "45/135 PARA": "PARA 45/135",
}

# Notebook orientation_mode (1-4) → canonical label
ORIENTATION_MODE_MAP = {
    1: "PERP 0/90",
    2: "PARA 0/90",
    3: "PERP 45/135",
    4: "PARA 45/135",
}

# Run-period specific sign overrides (only cases we have evidence for)
SIGN_OVERRIDES = {
    ("2023", "PERP 0/90"): -1,  # Spring 2023 PERP 0/90 uses opposite energy sign
}


def get_sign(value):
    if value>0:
        return 1
    elif value<0:
        return -1
    else:
        return 0


def normalize_orientation(label_or_mode):
    """Return canonical orientation label for a variety of inputs."""
    if isinstance(label_or_mode, int):
        if label_or_mode not in ORIENTATION_MODE_MAP:
            raise ValueError(f"Unsupported orientation mode: {label_or_mode}")
        return ORIENTATION_MODE_MAP[label_or_mode]
    if label_or_mode in ORIENTATION_TO_PHI:
        return label_or_mode
    if label_or_mode in ORIENTATION_ALIASES:
        return ORIENTATION_ALIASES[label_or_mode]
    raise ValueError(f"Unsupported orientation label: {label_or_mode}")


def infer_nudge_direction_from_sizes(orientation_label, pitch_size, yaw_size):
    """Infer the base nudge direction (before run-period overrides) from pitch/yaw sizes."""
    orientation_label = normalize_orientation(orientation_label)
    if orientation_label == "PERP 0/90":
        dir_val = -get_sign(pitch_size)
    elif orientation_label == "PARA 0/90":
        dir_val = -get_sign(yaw_size)
    elif orientation_label == "PERP 45/135":
        dir_val = -get_sign(yaw_size) if abs(yaw_size) >= abs(pitch_size) else get_sign(pitch_size)
    elif orientation_label == "PARA 45/135":
        dir_val = -get_sign(yaw_size) if abs(yaw_size) >= abs(pitch_size) else -get_sign(pitch_size)
    else:
        raise ValueError(f"Unsupported orientation label: {orientation_label}")
    return dir_val or -1


def direction_to_pitch_yaw(run_period, orientation_label, nudge_dir, base_step):
    """Map an abstract nudge direction to signed pitch/yaw steps and an energy sign."""
    orientation_label = normalize_orientation(orientation_label)
    energy_override = SIGN_OVERRIDES.get((run_period, orientation_label), 1)

    if orientation_label == "AMORPHOUS":
        pitch_delta = 0.0
        yaw_delta = 0.0
        energy_dir = 0.0
        return pitch_delta, yaw_delta, energy_dir
    if orientation_label == "PERP 0/90":
        pitch_delta = -nudge_dir * base_step
        yaw_delta = 0.0
    elif orientation_label == "PARA 0/90":
        pitch_delta = 0.0
        yaw_delta = -nudge_dir * base_step
    elif orientation_label == "PERP 45/135":
        pitch_delta = nudge_dir * base_step / (2 ** 0.5)
        yaw_delta = -nudge_dir * base_step / (2 ** 0.5)
    elif orientation_label == "PARA 45/135":
        pitch_delta = -nudge_dir * base_step / (2 ** 0.5)
        yaw_delta = -nudge_dir * base_step / (2 ** 0.5)
    else:
        raise ValueError(f"Unsupported orientation label: {orientation_label}")

    energy_dir = nudge_dir * energy_override
    return pitch_delta, yaw_delta, energy_dir

def delta_c_from_pitch_yaw(
    delta_h_deg, #pitch
    delta_v_deg, #yaw
    phi_deg,
    beam_pitch_deg=0.0,
    beam_yaw_deg=0.0
):
    delta_h_rad = np.deg2rad(delta_h_deg)
    delta_v_rad = np.deg2rad(delta_v_deg)    
    phi_rad = np.deg2rad(phi_deg)    
    delta_c_rad = delta_v_rad * np.cos(phi_rad) + delta_h_rad * np.sin(phi_rad)
    return delta_c_rad

def delta_c_to_peak(delta_c_rad, E0, Ei):
    g = 2.
    k = 26.5601
    
    deltaE = (delta_c_rad * (E0 - Ei)**2 ) / (k/g + delta_c_rad * (E0 - Ei) )

    return deltaE


############################################
# GONIOMETER CLASS
############################################
class Goniometer:
    def __init__(self, run_period):

        # possible diamond orientations [should I include amorphous?] 
        self.orientations = ["AMORPHOUS","PERP 0/90","PARA 0/90","PERP 45/135","PARA 45/135"]

        self.current_orientation = "Undefined"
        self.current_set_yaw = 0
        self.current_set_pitch = 0
        self.current_set_roll = 0
        self.current_set_x = 0
        self.current_set_y = 0

        self.current_rbv_yaw = 0
        self.current_rbv_pitch = 0
        self.current_rbv_roll = 0
        self.current_rbv_x = 0
        self.current_rbv_y = 0

        # backlash tracking, will be a number between -2.1 and 2.1 
        self.pitch_recent_moves = 0
        # backlash tracking, will be a number between -4.1 and 4.1 
        self.yaw_recent_moves = 0

        self.current_diamond_pitch = 0
        self.current_diamond_yaw = 0

        # 2.1 millidegrees of backlash in pitch
        self.backlash_pitch = 2.1/1000.0
        # around 4.1 millidegrees of backlash in yaw
        self.backlash_yaw = 4.1/1000.0

        # set rough initial goniometer values for each orientation 
        self.orientation_x = [0, 0, 0, 0, 0]
        self.orientation_y = [0, 0, 0, 0, 0]

        # based on data from each run period
        if run_period=="2020":
            self.orientation_pitch = [0.0,0.39,-0.73,0.39,1.81]
            self.orientation_yaw = [0.0,1.4,2.4,0.73,0.84]
            self.orientation_roll = [0.0,-10.5,-10.5,34.5,34.5]
        elif run_period=="2023":
            self.orientation_pitch = [0.0,-0.66,0.33,-1.75,-0.28]
            self.orientation_yaw = [0.0,0.17,1.28,0.96,1.06]
            self.orientation_roll = [0.0,162,162,-153,-153]
        elif run_period=="2025":
            self.orientation_pitch = [0.0,1.68,0.59,0.46,0.46]
            self.orientation_yaw = [0.0,1.52,1.94,1.94,0.49]
            self.orientation_roll = [0.0,-16.6, -16.6, 28.4, 28.4]
        else:
            print("Run period",run_period,"not currently set up")
            exit(0)
    

    # change orientation, track direction for backlash accounting (done in change_diamond_angles)
    def change_orientation(self, new_orientation):

        orientation_index = self.orientations.index(new_orientation)

        initial_pitch = self.current_set_pitch 
        initial_yaw = self.current_set_yaw 
        initial_roll = self.current_set_roll

        # set angle values change immediately (could add time delayed rbv)
        self.current_set_pitch = self.orientation_pitch[orientation_index]
        self.current_set_yaw = self.orientation_yaw[orientation_index]
        self.current_set_roll = self.orientation_roll[orientation_index]

        # size and sign of angle change in each direction
        nudge_pitch = self.current_set_pitch - initial_pitch 
        nudge_yaw = self.current_set_yaw - initial_yaw 
        nudge_roll = self.current_set_roll - initial_roll

        # change the diamond angles, accounting for backlash (no roll, not needed for energy changes)
        self.change_diamond_angles("pitch", nudge_pitch)
        self.change_diamond_angles("yaw",nudge_yaw)

        # tell the motors to start moving (modify readback values)
        self.start_your_engines("pitch",nudge_pitch)
        self.start_your_engines("yaw",nudge_yaw)
        self.start_your_engines("roll",nudge_roll)
        


    # here we eventually want the readback value to step in time, for now, just immediately set it to the right value since we don't know the motor speeds 
    def start_your_engines(self, motor, signed_nudge_size):

        if motor=="pitch":
            motor_speed = 0.001 # totally guess that it is 1 millidegree a second
            time_to_finish = abs(signed_nudge_size)/motor_speed
            self.current_rbv_pitch += signed_nudge_size
            # maybe change diamond angles here too?

        if motor=="yaw":
            motor_speed = 0.001
            time_to_finish = abs(signed_nudge_size)/motor_speed 
            self.current_rbv_yaw += signed_nudge_size 

        if motor=="roll":
            motor_speed = 0.01
            time_to_finish = abs(signed_nudge_size)/motor_speed 
            self.current_rbv_roll += signed_nudge_size 

        if motor=="x":
            motor_speed = 0.5 
            time_to_finish = abs(signed_nudge_size)/motor_speed 
            self.current_rbv_x += signed_nudge_size 
            
        if motor=="y":
            motor_speed = 0.5 
            time_to_finish = abs(signed_nudge_size)/motor_speed 
            self.current_rbv_y += signed_nudge_size


    # this is where we account for backlash
    def change_diamond_angles(self, motor, signed_nudge_size):

        if abs(signed_nudge_size)<0.0001:
            #print("nudge of size less than a 1/10 millidegree given, ignoring")
            return

        if motor=="pitch":
                
            # nudge needs to be same sign as recent_moves and recent_moves needs to be +/- 2.1 for no backlash
            if get_sign(signed_nudge_size)*self.backlash_pitch!=self.pitch_recent_moves:
                # amount of backlash is bigger than the current nudge, so diamond angle won't change
                if abs(self.pitch_recent_moves+2.0*signed_nudge_size)<self.backlash_pitch:
                    # but progress is made towards clearing the backlash
                    self.pitch_recent_moves+=2.0*signed_nudge_size 
                else:
                    # have backlash that will be partially cancelled 
                    nudge_amount_cancelled = (self.backlash_pitch-abs(self.pitch_recent_moves))/2.0
                    actual_nudge_size_signed = signed_nudge_size-get_sign(signed_nudge_size)*nudge_amount_cancelled
                    self.current_diamond_pitch+=actual_nudge_size_signed
                    self.pitch_recent_moves = get_sign(actual_nudge_size_signed)*self.backlash_pitch
            else:
                # just change the diamond angle by the requested amount 
                self.pitch_recent_moves+=2.0*signed_nudge_size
                self.current_diamond_pitch+=signed_nudge_size
                if abs(self.pitch_recent_moves)>self.backlash_pitch:
                    # max size of recent moves should be max size of the backlash, but keep the correct sign
                    self.pitch_recent_moves = get_sign(self.pitch_recent_moves)*self.backlash_pitch
        
        elif motor == "yaw":
            if get_sign(signed_nudge_size)*self.backlash_yaw!=self.yaw_recent_moves:
                # amount of backlash is bigger than the current nudge, so diamond angle won't change
                if abs(self.yaw_recent_moves+2.0*signed_nudge_size)<self.backlash_yaw:
                    # but progress is made towards clearing the backlash
                    self.yaw_recent_moves+=2.0*signed_nudge_size 
                else:
                     # backlash should be cleared, but we will have a less effective nudge
                    nudge_amount_cancelled = (self.backlash_yaw-abs(self.yaw_recent_moves))/2.0
                    actual_nudge_size_signed = signed_nudge_size-get_sign(signed_nudge_size)*nudge_amount_cancelled
                    self.current_diamond_yaw+=actual_nudge_size_signed
                    self.yaw_recent_moves = get_sign(actual_nudge_size_signed)*self.backlash_yaw


            else:# no backlash
                self.yaw_recent_moves+=2.0*signed_nudge_size
                self.current_diamond_yaw+=signed_nudge_size

                if abs(self.yaw_recent_moves)>self.backlash_yaw:
                    self.yaw_recent_moves = get_sign(self.yaw_recent_moves)*self.backlash_yaw


    def do_nudge(self, motor, nudge_size_signed):
        if motor=="pitch":
            self.current_set_pitch += nudge_size_signed 
            self.change_diamond_angles(motor,nudge_size_signed)
            self.start_your_engines(motor,nudge_size_signed)

        elif motor=="yaw":
            self.current_set_yaw += nudge_size_signed
            self.change_diamond_angles(motor,nudge_size_signed)
            self.start_your_engines(motor,nudge_size_signed)


    def print_state(self):
        print("current set pitch is",self.current_set_pitch)
        print("current set yaw is",self.current_set_yaw)
        print("current set roll is",self.current_set_roll)

        print("current diamond pitch is",self.current_diamond_pitch)
        print("current diamond yaw is",self.current_diamond_yaw)

        print("current pitch recent moves is",self.pitch_recent_moves)
        print("current yaw recent moves is",self.yaw_recent_moves)


    def return_state(self):
        return {"pitch_set":self.current_set_pitch, "yaw_set":self.current_set_yaw,  "diamond_pitch":self.current_diamond_pitch, "diamond_yaw":self.current_diamond_yaw}
            
    def return_diamond_pitch(self):
        return self.current_diamond_pitch
    
    def return_diamond_yaw(self):
        return self.current_diamond_yaw
    
    def return_set_pitch(self):
        return self.current_set_pitch
    
    def return_set_yaw(self):
        return self.current_set_yaw
    
    def return_set_roll(self):
        return self.current_set_roll

class BeamState:
    def __init__(self, beam_pitch_deg=0.0, beam_yaw_deg=0.0):
        self.beam_pitch_deg = beam_pitch_deg
        self.beam_yaw_deg = beam_yaw_deg


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
    def __init__(self, base_peak_position, dose_slope, beam_energy_E0, coherent_edge_Ei):
        self.base_peak_position = base_peak_position
        self.dose_slope = dose_slope
        self.E0 = beam_energy_E0
        self.Ei = coherent_edge_Ei

        self.current_peak_position = float(base_peak_position)

    def reset(self):
        self.current_peak_position = float(self.base_peak_position)

    def update(self, delta_c_rad, dose):

        deltaE_step = delta_c_to_peak(delta_c_rad, self.E0, self.Ei)

        dose_shift = self.dose_slope * dose

        self.current_peak_position = self.base_peak_position + dose_shift + (self.current_peak_position - self.base_peak_position) + deltaE_step

        return self.current_peak_position


# ============================================================
# HIGH-LEVEL SIMULATOR
# ============================================================

class CoherentBremsstrahlungSimulator:
    def __init__(
        self,
        base_peak_position,
        dose_slope,
        beam_energy_E0,
        coherent_edge_Ei,
        orientation,
        run_period="2020",
        use_streamlined_energy=False,
        nudge_energy_size_pitch=10.0,
        nudge_energy_size_yaw=10.0,
        latency_setpoint_to_readback=8,
    ):
        self.run_period = run_period
        self.beam_state = BeamState()
        self.orientation = normalize_orientation(orientation)

        if self.orientation == "AMORPHOUS":
            base_peak_position = 0.0
            coherent_edge_Ei = 0.0

        self.goni = Goniometer(run_period=run_period)
        self.goni.change_orientation(self.orientation)

        self.phi_deg = ORIENTATION_TO_PHI[self.orientation]

        self.dose = DiamondDose()
        self.peak = CoherentPeakTracker(
            base_peak_position=base_peak_position,
            dose_slope=dose_slope,
            beam_energy_E0=beam_energy_E0,
            coherent_edge_Ei=coherent_edge_Ei
        )

        # When enabled, mimic the streamlined notebook's heuristic energy update
        self.use_streamlined_energy = use_streamlined_energy
        self.nudge_energy_size_pitch = nudge_energy_size_pitch
        self.nudge_energy_size_yaw = nudge_energy_size_yaw
        self.current_peak_streamlined = float(base_peak_position)

        # Latency (in steps/seconds) before a commanded motion affects the peak
        self.latency_steps = max(0, int(round(latency_setpoint_to_readback)))
        self._pending = []  # list of (due_step, pitch_true, yaw_true, pitch_set, yaw_set)
        self._t = 0



    def _streamlined_energy_step(self, pitch_set_change_deg, yaw_set_change_deg):
        """Heuristic energy update used by the streamlined notebook."""
        energy_change = 0.0

        if self.orientation == "AMORPHOUS":
            self.current_peak_streamlined = 0.0
            return 0.0, 0.0

        def infer_nudge_dir():
            """Reconstruct nudge_direction sign from setpoint deltas and orientation mapping used in the streamlined notebook."""
            # Orientation labels match ORIENTATION_TO_PHI keys
            label = self.orientation
            if label in ("PERP 0/90", "PARA 0/90"):
                # Notebook orientation_mode 1/2: sign is opposite of the setpoint change (pitch for PERP, yaw for PARA)
                if label == "PERP 0/90" and abs(pitch_set_change_deg) > 0:
                    return -get_sign(pitch_set_change_deg)
                if label == "PARA 0/90" and abs(yaw_set_change_deg) > 0:
                    return -get_sign(yaw_set_change_deg)
            if label == "PERP 45/135":
                # Notebook orientation_mode 3: pitch increases with +dir, yaw decreases with +dir
                if abs(pitch_set_change_deg) >= abs(yaw_set_change_deg):
                    return get_sign(pitch_set_change_deg)
                return -get_sign(yaw_set_change_deg)
            if label == "PARA 45/135":
                # Notebook orientation_mode 4: both pitch and yaw decrease with +dir
                if abs(pitch_set_change_deg) >= abs(yaw_set_change_deg):
                    return -get_sign(pitch_set_change_deg)
                return -get_sign(yaw_set_change_deg)
            # Fallback to dominant component sign
            dominant = pitch_set_change_deg if abs(pitch_set_change_deg) >= abs(yaw_set_change_deg) else yaw_set_change_deg
            return get_sign(dominant)

        if abs(pitch_set_change_deg) > 0.0 or abs(yaw_set_change_deg) > 0.0:
            nudge_dir = infer_nudge_dir() * SIGN_OVERRIDES.get((self.run_period, self.orientation), 1)
            energy_change = (
                (pitch_set_change_deg ** 2 * self.nudge_energy_size_pitch ** 2 +
                 yaw_set_change_deg ** 2 * self.nudge_energy_size_yaw ** 2) ** 0.5
            ) / 0.001
            energy_change *= nudge_dir

        self.current_peak_streamlined += energy_change
        return energy_change, self.current_peak_streamlined


    def _pop_due_moves(self):
        """Return accumulated deltas whose due_step has arrived."""
        due_true_pitch = 0.0
        due_true_yaw = 0.0
        due_set_pitch = 0.0
        due_set_yaw = 0.0
        remaining = []
        for due_step, tp, ty, sp, sy in self._pending:
            if due_step <= self._t:
                due_true_pitch += tp
                due_true_yaw += ty
                due_set_pitch += sp
                due_set_yaw += sy
            else:
                remaining.append((due_step, tp, ty, sp, sy))
        self._pending = remaining
        return due_true_pitch, due_true_yaw, due_set_pitch, due_set_yaw


    def step(self, dpitch_deg, dyaw_deg, delta_dose):
        """
        Apply a goniometer move and dose increment.

        Returns:
            delta_c_deg, peak_position
        """
        prev_true_pitch = self.goni.return_diamond_pitch()
        prev_true_yaw = self.goni.return_diamond_yaw()
        prev_set_pitch = self.goni.return_set_pitch()
        prev_set_yaw = self.goni.return_set_yaw()

        self.goni.do_nudge("pitch", dpitch_deg)
        self.goni.do_nudge("yaw", dyaw_deg)

        curr_true_pitch = self.goni.return_diamond_pitch()
        curr_true_yaw = self.goni.return_diamond_yaw()
        curr_set_pitch = self.goni.return_set_pitch()
        curr_set_yaw = self.goni.return_set_yaw()

        pitch_true_change_deg = curr_true_pitch - prev_true_pitch
        yaw_true_change_deg = curr_true_yaw - prev_true_yaw
        pitch_set_change_deg = curr_set_pitch - prev_set_pitch
        yaw_set_change_deg = curr_set_yaw - prev_set_yaw

        self.dose.add(delta_dose)

        # enqueue motion; it will affect peak after latency
        if self.latency_steps == 0:
            due_true_pitch = pitch_true_change_deg
            due_true_yaw = yaw_true_change_deg
            due_set_pitch = pitch_set_change_deg
            due_set_yaw = yaw_set_change_deg
        else:
            due_step = self._t + self.latency_steps
            self._pending.append((due_step, pitch_true_change_deg, yaw_true_change_deg, pitch_set_change_deg, yaw_set_change_deg))
            due_true_pitch, due_true_yaw, due_set_pitch, due_set_yaw = self._pop_due_moves()

        if self.orientation == "AMORPHOUS":
            delta_c = 0.0
            peak = 0.0
        elif self.use_streamlined_energy:
            # Match streamlined notebook: use set-point deltas, no phi projection, no dose term
            delta_c = 0.0
            _, peak = self._streamlined_energy_step(due_set_pitch, due_set_yaw)
        else:
            delta_c = delta_c_from_pitch_yaw(
                delta_h_deg=due_true_pitch,
                delta_v_deg=due_true_yaw,
                phi_deg=self.phi_deg,
                beam_pitch_deg=self.beam_state.beam_pitch_deg,
                beam_yaw_deg=self.beam_state.beam_yaw_deg
            )

            peak = self.peak.update(delta_c_rad=delta_c, dose=self.dose.dose)

        self._t += 1
        return delta_c, peak
    

# ============================================================
# MAIN SIMULATION
# ============================================================

def main():

    sim = CoherentBremsstrahlungSimulator(
        base_peak_position=8600,   # MeV
        dose_slope=0.05,           # MeV / dose
        beam_energy_E0=11600.0,    # MeV
        coherent_edge_Ei=8600.0,   # MeV
        orientation="PARA 0/90"
    )

    # Beam misalignment
    sim.beam_state.beam_pitch_deg = 0.03
    sim.beam_state.beam_yaw_deg = -0.02

    print("Step | Pitch(deg) | Yaw(deg) | Dose | Delta c | Peak")
    print("-" * 65)

    for i in range(10):
        delta_c, peak = sim.step(
            dpitch_deg=+0.001,
            dyaw_deg=+0.001,
            delta_dose=0.0
        )

        print(
            f"{i:4d} | "
            f"{sim.goni.return_diamond_pitch():10.4f} | "
            f"{sim.goni.return_diamond_yaw():8.4f} | "
            f"{sim.dose.dose:4.1f} | "
            f"{delta_c:7.7f} | "
            f"{peak:7.3f}"
        )

        
if __name__ == "__main__":
    main()
