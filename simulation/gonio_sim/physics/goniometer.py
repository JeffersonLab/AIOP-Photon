import numpy as np
from gonio_sim.physics.root_io import init_root

def compute_goniometer_angles(phi, c, i, offsets):
    phi0, Bv, Bh, Theta, Phi = (
        offsets["phi0"], offsets["Bv"], offsets["Bh"], offsets["Theta"], offsets["Phi"]
    )
    Ga = phi - phi0
    Gv = c*np.cos(phi) - i*np.sin(phi) - Theta*np.cos(Ga + Phi) + Bv
    Gh = c*np.sin(phi) + i*np.cos(phi) - Theta*np.sin(Ga + Phi) + Bh
    return Ga, Gv, Gh




def compute_peak_energy(params):
    (
        i, total,
        yaw_true_nowobble, pitch_true_nowobble,
        yaw_true, pitch_true,
        yaw_readback, pitch_readback,
        beam_delh, beam_delv,
        base_args, dose, damage, offsets,
    ) = params
    

    ROOT = init_root(base_args)


    h_incoh = ROOT.amorph_intensity(
        base_args["ebeam"], base_args["ibeam"],
        base_args["peresol"], base_args["penergy0"], base_args["penergy1"]
    )[0]
    h_incoh.SetDirectory(0)


    thetav_eff = yaw_true + beam_delh
    thetah_eff = pitch_true + beam_delv
    base_args["thetah"], base_args["thetav"] = thetah_eff, thetav_eff
    
    beamx = np.random.normal(base_args["xoffset"], base_args["beamx_noise"])
    beamy = np.random.normal(base_args["yoffset"], base_args["beamy_noise"])
    base_args["xoffset"], base_args["yoffset"] = beamx, beamy
    

    hlist = ROOT.cobrems_intensity(
        base_args["radname"], base_args["iradview"],
        base_args["ebeam"], base_args["ibeam"],
        base_args["xyresol"], base_args["thetah"], base_args["thetav"],
        base_args["xoffset"], base_args["yoffset"], base_args["phideg"],
        base_args["xsigma"], base_args["ysigma"], base_args["xycorr"],
        base_args["peresol"], base_args["penergy0"], base_args["penergy1"], 0
    )
    h_coh = hlist[0]
    h_coh.SetDirectory(0)

    n_bins = h_coh.GetNbinsX()
    energy = np.array([h_coh.GetBinCenter(b) for b in range(1, n_bins + 1)])
    y_coh = np.array([h_coh.GetBinContent(b) for b in range(1, n_bins + 1)])
    y_inc = np.array([h_incoh.GetBinContent(b) for b in range(1, n_bins + 1)])
    
    # Convert to MeV
    E_MeV = 1000.0 * energy

    # Apply diamond degradation
    y_coh = damage.apply(E_MeV, y_coh, dose)

    # Compute enhancement and peak
    enhancement = np.divide(y_coh, y_inc, out=np.zeros_like(y_coh), where=y_inc != 0)
    peak_energy = float(energy[np.nanargmax(enhancement)])

    phi = np.deg2rad(base_args["phideg"])
    c, i_angle = pitch_true * 1e-3, yaw_true * 1e-3
    Ga, Gv, Gh = compute_goniometer_angles(phi, c, i_angle, offsets)
    
    return (
        yaw_readback, pitch_readback, yaw_true_nowobble, pitch_true_nowobble, yaw_true, pitch_true,
        beam_delh, beam_delv,
        thetah_eff, thetav_eff,
        beamx, beamy, peak_energy,
        np.rad2deg(Ga), np.rad2deg(Gv), np.rad2deg(Gh)
    )




def get_sign(value):
    if value>0:
        return 1
    elif value<0:
        return -1
    else:
        return 0


# goniometer doesn't know about the photon beam, but does know about backlash
# need separate function to change photon energy based on actual change in diamond angles after backlash

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