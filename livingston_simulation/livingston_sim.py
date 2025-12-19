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


############################################
# GONIOMETER CLASS
############################################
class Goniometer:
    def __init__(self, run_period):

        # possible diamond orientations [should I include amorphous?] 
        self.orientations = ["0/90 PERP","0/90 PARA","45/135 PERP","45/135 PARA"]

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
        self.orientation_x = [0, 0, 0, 0]
        self.orientation_y = [0, 0, 0, 0]

        # based on data from each run period
        if run_period=="2020":
            self.orientation_pitch = [0.39,-0.73,0.39,1.81]
            self.orientation_yaw = [1.4,2.4,0.73,0.84]
            self.orientation_roll = [-10.5,-10.5,34.5,34.5]
        elif run_period=="2023":
            self.orientation_pitch = [-0.66,0.33,-1.75,-0.28]
            self.orientation_yaw = [0.17,1.28,0.96,1.06]
            self.orientation_roll = [162,162,-153,-153]
        elif run_period=="2025":
            self.orientation_pitch = [1.68,0.59,0.46,0.46]
            self.orientation_yaw = [1.52,1.94,1.94,0.49]
            self.orientation_roll = [-16.6, -16.6, 28.4, 28.4]
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
            print("nudge of size less than a 1/10 millidegree given, ignoring")
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

############################################
# GONIOMETER CLASS
############################################

class BeamState:
    """
    Tracks beam alignment.
    """

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
        orientation=CrystalOrientation.PARA_0_90,
        run_period="2020"
    ):
        self.beam_state = BeamState()
        self.orientation = orientation

        self.goni = Goniometer(run_period="2020")
        self.goni.change_orientation(orientation)

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
        
        prev_true_pitch = self.goni.return_diamond_pitch()
        prev_true_yaw = self.goni.return_diamond_yaw()

        self.goni.do_nudge("pitch", dpitch_deg)
        self.goni.do_nudge("yaw", dyaw_deg)

        curr_true_pitch = self.goni.return_diamond_pitch()
        curr_true_yaw = self.goni.return_diamond_yaw()

        pitch_true_change_deg = curr_true_pitch - prev_true_pitch
        yaw_true_change_deg = curr_true_yaw - prev_true_yaw

        self.dose.add(delta_dose)

        delta_c = delta_c_from_pitch_yaw(
            pitch_change_deg=pitch_true_change_deg,
            yaw_change_deg=yaw_true_change_deg,
            orientation=self.orientation,
            beam_pitch_deg=self.beam_state.beam_pitch_deg,
            beam_yaw_deg=self.beam_state.beam_yaw_deg
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
    sim.beam_state.beam_pitch_deg = 0.03
    sim.beam_state.beam_yaw_deg = -0.02

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
