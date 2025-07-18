

def clean_file(inputFilePath, ColumnName, outputFilePath, badValues=["<<","Network","disconnection",">>","<"]):

    print("starting file",inputFilePath)
    # need to replace things like
    # << Network disconnection >>,
    # with nan.
    # for run number, we may be able to use ccdb to recover some of the info.

    newFile = "Date-Time, "+ColumnName+"\n"

    last_time = "ooooooSpoookyOOOOOO"

    with open(inputFilePath,"r") as file:

        line_list = file.readlines()

        for line in line_list:

            # element zero is date, 1 is time, 3 is column from EPICS (assuming no spaces)
            split_line = line.split()


            # catches "<< Network disconnection >>", and can add bad vals (ie like run number of 0)
            if split_line[2] in badValues:
                print(split_line)
                split_line[2] = "Undefined"

                continue # for now just skip if it is undefined or 0

            output_time = split_line[0]+"T"+split_line[1]
            # convert split_line[0] and split_line[1] to a python datetime friendly format for later
            #output_line = output_time+","+split_line[2]# this breaks radiator name, which has spaces in the actual variable values, using [2:] should work
            output_line = output_time+","+" ".join(split_line[2:])
            # some readback variables have multiple readings for each second
            # don't have time precision to keep all readings, so just take first in those cases
            if output_time==last_time:
                continue
            else:
                last_time = output_time

            newFile += output_line+"\n"


    with open(outputFilePath,"w") as file:
        file.write(newFile)

    print("successfully wrote file",outputFilePath)


# takes the result of one myget command in a .txt file
# format Date-Time Column-Info
# and converted Date-Time to format that is python datetime friendly
# and cleans obvious bad values in Column
# then outputs as a csv file with two columns, datetime and column
# these will later be merged into a single csv
if __name__ == "__main__":

    run_periods = ["Spring2020","Spring2023","Spring2025"]

    for runPeriod in run_periods:
        # all run configuration info
        clean_file("rawData/"+runPeriod+"/cbrem_plane.txt","CBREM:PLANE","csv_data/initial_converted/"+runPeriod+"/cbrem_plane.csv")
        clean_file("rawData/"+runPeriod+"/daq_status.txt","DAQ:STATUS","csv_data/initial_converted/"+runPeriod+"/daq_status.csv")
        clean_file("rawData/"+runPeriod+"/goni_pitch_rbv.txt","GONI:PITCH.RBV","csv_data/initial_converted/"+runPeriod+"/goni_pitch_rbv.csv")
        clean_file("rawData/"+runPeriod+"/goni_pitch.txt","GONI:PITCH","csv_data/initial_converted/"+runPeriod+"/goni_pitch.csv")
        clean_file("rawData/"+runPeriod+"/goni_roll_rbv.txt","GONI:ROLL.RBV","csv_data/initial_converted/"+runPeriod+"/goni_roll_rbv.csv")
        clean_file("rawData/"+runPeriod+"/goni_roll.txt","GONI:ROLL","csv_data/initial_converted/"+runPeriod+"/goni_roll.csv")
        clean_file("rawData/"+runPeriod+"/goni_x_rbv.txt","GONI:X.RBV","csv_data/initial_converted/"+runPeriod+"/goni_x_rbv.csv")
        clean_file("rawData/"+runPeriod+"/goni_x.txt","GONI:X","csv_data/initial_converted/"+runPeriod+"/goni_x.csv")
        clean_file("rawData/"+runPeriod+"/goni_y_rbv.txt","GONI:Y.RBV","csv_data/initial_converted/"+runPeriod+"/goni_y_rbv.csv")
        clean_file("rawData/"+runPeriod+"/goni_y.txt","GONI:Y","csv_data/initial_converted/"+runPeriod+"/goni_y.csv")
        clean_file("rawData/"+runPeriod+"/goni_yaw_rbv.txt","GONI:YAW.RBV","csv_data/initial_converted/"+runPeriod+"/goni_yaw_rbv.csv")
        clean_file("rawData/"+runPeriod+"/goni_yaw.txt","GONI:YAW","csv_data/initial_converted/"+runPeriod+"/goni_yaw.csv")
        clean_file("rawData/"+runPeriod+"/run_number.txt","RunNumber","csv_data/initial_converted/"+runPeriod+"/run_number.csv",["<<","0"])


        # photon and electron beam info
        clean_file("rawData/"+runPeriod+"/ac_x.txt","AC:X","csv_data/initial_converted/"+runPeriod+"/ac_x.csv")
        clean_file("rawData/"+runPeriod+"/ac_y.txt","AC:Y","csv_data/initial_converted/"+runPeriod+"/ac_y.csv")
        clean_file("rawData/"+runPeriod+"/ebeam_current.txt","EBEAM:CURRENT","csv_data/initial_converted/"+runPeriod+"/ebeam_current.csv")
        clean_file("rawData/"+runPeriod+"/ebeam_energy.txt","EBEAM:ENERGY","csv_data/initial_converted/"+runPeriod+"/ebeam_energy.csv")
        clean_file("rawData/"+runPeriod+"/ebeam_x.txt","EBEAM:X","csv_data/initial_converted/"+runPeriod+"/ebeam_x.csv")
        clean_file("rawData/"+runPeriod+"/ebeam_y.txt","EBEAM:Y","csv_data/initial_converted/"+runPeriod+"/ebeam_y.csv")
        clean_file("rawData/"+runPeriod+"/photonbeam_energy_desired.txt","PBEAM:SET","csv_data/initial_converted/"+runPeriod+"/photonbeam_energy_desired.csv")
        clean_file("rawData/"+runPeriod+"/photonbeam_energy_uncertainty.txt","PBEAM:WIDTH","csv_data/initial_converted/"+runPeriod+"/photonbeam_energy_uncertainty.csv")
        clean_file("rawData/"+runPeriod+"/photonbeam_energy.txt","PBEAM:ENERGY","csv_data/initial_converted/"+runPeriod+"/photonbeam_energy.csv")
        clean_file("rawData/"+runPeriod+"/radiator_id.txt","RADIATOR:ID","csv_data/initial_converted/"+runPeriod+"/radiator_id.csv")
        clean_file("rawData/"+runPeriod+"/radiator_name.txt","RADIATOR:NAME","csv_data/initial_converted/"+runPeriod+"/radiator_name.csv")
        if runPeriod=='Spring2023' or runPeriod=='Spring2025':
            clean_file("rawData/"+runPeriod+"/beam_lock.txt","AC:X","csv_data/initial_converted/"+runPeriod+"/beam_lock.csv")
