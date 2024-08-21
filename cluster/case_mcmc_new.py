import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy 
import os
from exh_functions import *
from exh_processing import *
from default_configs import *
from os import makedirs

#Whenever you want to submit a job, you should specify how many runs you want (accepted models) and 
#the output folder name
created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws

label = generate_unique_label() #we generate a unique label to identify the outputs of this particular run
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv" #load the initial exhumation values of the model
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/" #stores accepted and rejected parameters
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/" #stores accepted and rejected exhumation outputs
model_histories_folder = f"{output_folder}/{model_name}/model_histories/{args.folder}/" #stores accepted and rejected model histories
makedirs(model_params_folder,exist_ok=True)
makedirs(model_samples_folder,exist_ok=True)
makedirs(model_histories_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history_samples}")
print(f"[{time_string()}] {'Simulating based on observation':<40} {samples}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()

#LOAD NODDY MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history_samples, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)
hist = pynoddy.history.NoddyHistory(history_samples)
hist.change_cube_size(cubesize)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)

samples = pd.read_csv(samples,delimiter = ',')

#DEFINE IMPORTANT VALUES
og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']  
        og_depths.append(z)

samples_z = []
for i in range(len(samples)):
    z = samples.iloc[i]['Z']
    samples_z.append(z)

 #CALCULATING ORIGINAL EXHUMATION
print(f"[{time_string()}] Calculating original exhumation")

if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
  
else:
    print(f"[{time_string()}] Calculating from scratch.")
    all_liths = np.arange(11,21)
    _,samples_noddy_pos,_ = calc_new_position(hist, og_depths, og_depths, all_liths,samples,label)   
    diff = [x - y for x, y in zip(samples_noddy_pos, samples_z)]
    current_exhumation = [x - y - z for x,y,z in zip(samples_noddy_pos, diff, og_depths)]
    samples['exhumation'] = current_exhumation
    samples['respected'] = 0
diff = [diff[i] for i in sample_num]
og_depths = [og_depths[i] for i in sample_num]
samples = samples.iloc[sample_num]
#current_exhumation = samples.loc[sample_num].copy()

og_params = []
for i in event:
    event_data = [i]
    for j, props in enumerate(prop):
        propert = hist.events[i].properties[props]
        event_data.append(propert)
    og_params.append(event_data)
    
og_params_df = pd.DataFrame(og_params, columns = ['event','Z','Slip'])

current_exhumation.reset_index(drop=True,inplace=True)

#SIMULATION
accepted = 0
total_runs = 0
current_params = og_params
target_rate = 0.4
#STORAGE
score = []
accepted_params = pd.DataFrame(columns = ['Event'] + prop + ['n_draw'])
accepted_exhumation = []
rejected_params = pd.DataFrame(columns = ['Event'] + prop + ['n_draw'])

print(f"[{time_string()}] Starting MCMC")
for i in range(n_draws):

    while accepted < n_draws:
        proposed_hist = copy.deepcopy(hist)

        #propose new starting parameters
        proposed_params, proposed_params_df = disturb_property(proposed_hist,event,prop,std)
        try:
            proposed_exhumation,_,new_hist = calc_new_position(proposed_hist, diff, 
                                                      og_depths,lith_list, samples.copy(),label)
        except IndexError: #sometimes, though not often, there is an IndexError because of a Noddy resolution error.
            continue
        proposed_exhumation.reset_index(drop=True,inplace=True)    
        
        #calculate likelihood and priors
        current_likelihood,current_score,current_samples = likelihood_and_score(current_exhumation)
        proposed_likelihood,proposed_score,proposed_samples = likelihood_and_score(proposed_exhumation)
        current_prior = prior_dist(og_params, current_params, std)
        proposed_prior = prior_dist(og_params, proposed_params, std)
        print(current_likelihood, proposed_likelihood, current_prior, proposed_prior)

        print(f"Model score: {proposed_score}")
        print(f"proposed exhumation {proposed_exhumation['exhumation']}")
        
        #accept or reject
        #acceptance_ratio = (proposed_prior * proposed_likelihood) / (current_prior * current_likelihood)
        #acceptance_ratio_log = (np.log(proposed_prior)+np.log(proposed_likelihood)) - (np.log(current_prior)+np.log(current_likelihood))
        
        prior_ratio = np.log(proposed_prior) - np.log(current_prior)
        likelihood_ratio = np.log(proposed_likelihood) - np.log(current_likelihood)
        balance_factor = 0.3
        acceptance_ratio_log = balance_factor * prior_ratio + (1 - balance_factor) * likelihood_ratio    
        acceptance_ratio = np.exp(acceptance_ratio_log)
        print(f"Acceptance ratio: {acceptance_ratio}")

        #Adaptive threshold depending on acceptance rate:
        if total_runs > 0:
            acceptance_rate = accepted / total_runs
            if acceptance_rate < target_rate:
                threshold = np.random.rand(1)
            else:
                threshold = 1.0
        else:
            threshold = np.random.rand(1)
        
        #random_n = np.random.rand(1)
        print(f"Threshold: {threshold}")

        if acceptance_ratio > threshold:
            current_params_df = proposed_params_df
            current_params = proposed_params
            current_exhumation = proposed_exhumation
            hist = proposed_hist
            accepted += 1
            print(f"accepted model number {accepted}")

            #store stuff for each run
            np.save(f"{model_params_folder}/accepted_params_{label}_draw{total_runs}.npy", current_params)
            acc_hist = f'{model_histories_folder}/acc_hist_{label}_draw{total_runs}.his'
            os.rename(new_hist, acc_hist)
            
            #store stuff for later
            score.append([proposed_score, i])
            accepted_params = pd.concat([accepted_params, current_params_df], ignore_index=True)
            accepted_exhumation.append(current_exhumation['exhumation'])
        else:
            np.save(f"{model_params_folder}/rejected_params_{label}_draw{total_runs}.npy", proposed_params)
            rejected_params = pd.concat([rejected_params, proposed_params_df], ignore_index=True)
            rej_hist = f'{model_histories_folder}/rej_hist_{label}_draw{total_runs}.his'
            os.rename(new_hist, rej_hist)

        total_runs += 1
        print(f"Total runs: {total_runs}")

print(f"The acceptance rate was: {accepted / total_runs}")
#SAVE THE STUFF TO THE CLUSTER
print(f"[{time_string()}] Saving all the important files")
scores = pd.DataFrame(score, columns = ['score', 'iteration'])
accepted_params.to_csv(f"{model_params_folder}/acc_params_{label}.csv", index = False)
rejected_params.to_csv(f"{model_params_folder}/rej_params_{label}.csv", index = False)
np.save(f"{model_params_folder}/accepted_exhumation_{label}.npy", accepted_exhumation)

print(f"[{time_string()}] Complete")
clean(label)

