This repo has the code that is currently being used for my portion of the AIOP project at Jefferson Lab. The AIOP_CoherentEdgePosition.pdf document has a more detailed summary of current results. 

The following outlines the steps to use the code. 

1) The .txt files in myGetCommands are the commands I use to access the archived EPICS data for the Spring 2020 run period. These must be run on one of the gluon machines. gluons150-155 are for general use. Instructions on accessing the gluons is given in Appendix C of the pdf.

2) cleanAndConvertData_txt_to_csv.py converts the resulting .txt files to csv files. It creates a .csv file with two variables, the new setpoint of the variable of interest, as well as the Date-Time when the new value was set. 

3) merge_all_csv.ipynb is used to merge the resulting csv files for all the variables. This is where the majority of the data manipulation occurs, including fixing the run numbers and defining the nudge sequences. 

4) add_beam_up_time_combined.ipynb adds a variable for the amount of time since the last beam drop occurred. This is for use in nonudge-study_combined.ipynb, so information about nudges are removed to save space. 

The remaining notebooks are used to visualize the output of the .csv files produced by the steps above. 
