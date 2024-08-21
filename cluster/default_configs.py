import numpy as np
## This is where we store the input data and files for each simulation. Comment out the input data when not used.
#### General
cubesize = 100
noddy_exe = "/rwthfs/rz/cluster/home/ho640525/projects/pynoddy/pynoddy/noddyapp/noddy"
history = 'data/input_files/bregenz_samples.his'
samples = 'data/input_files/bregenz_data.csv'
output_folder = "data/outputs"
model_name = "bregenz"
save_each = False #save output of each run (recommended)
save_overall = True #save output after all runs are finished 

#### Inputs for sensitivity study
#sample_num = [4]
#lith_list = [15]
#event = [22]
#prop = ['Dip', 'Dip Direction', 'Pitch']

#### Inputs for MCMC study 
prop = ['Z','Slip']
event = [21,22,24]
std = [300,800] 
lith_list = [13,15,17]
sample_num = [2,4,6]

#### Inputs for MCMC study using synthetic measurements
sigma = 800
synthetic_data = [[21, 3400],[22, 3300],[24,5400]]
