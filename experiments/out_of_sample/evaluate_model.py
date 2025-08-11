import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard, to_standard
# from diffusion_for_fusion.evaluate_configuration_sheet_curent import evaluate_configuration as evaluate_configuration_sheet
from diffusion_for_fusion.evaluate_configuration_vmec import evaluate_configuration as evaluate_configuration_vmec
from experiments.conditional_diffusion.load_quasr_data import prepare_data_from_config
import os
import time
import sys


"""
Generate data out-of-sample using one of the conditions. 

Run with,
    python evaluate_model.py <condition_index>
"""

condition_options = [
    [0.36, 4.5, 2, 0], # QA, nfp=2
    [0.5, 18.5, 3, 0],  # QA, nfp=3
    [0.5, 9.0, 3, 1],  # QH, nfp=3
    [1.4, 11.0, 4, 1],  # QH, nfp=4
    [2.5, 17.0, 5, 1], # QH, nfp=5
    [2.0, 14.0, 6, 1], # QH, nfp=6
    [3.7, 11.0, 7, 1], # QH, nfp=7
    [3.5, 22.0, 8, 1] # QH, nfp=8
    ]

# choose one of the conditions from list
idx_condition = int(sys.argv[1])

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "../conditional_diffusion/output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"

# number of samples
n_samples = 10

""" Load up the model"""

config_pickle = indir+"config.pickle"
model_path = indir+"model.pth"

# load the data
data = pickle.load(open(config_pickle, "rb"))
config = data['config']
input_dim = data['input_dim']
state_dict = torch.load(model_path)

# init the model
diffusion, model = init_conditional_diffusion_model_from_config(config, input_dim)
diffusion.load_state_dict(state_dict)
diffusion.eval()  # Set to evaluation mode
model.eval()  # Set to evaluation mode

# load, PCA, and standardize data
X_train, X_mean, X_std, Y_train, Y_mean, Y_std, pca = prepare_data_from_config(config)


""" Prepare the conditions """

# get the condition
iota_condition = condition_options[idx_condition][0]
aspect_ratio_condition = condition_options[idx_condition][1]
nfp_condition = condition_options[idx_condition][2]
helicity_condition = condition_options[idx_condition][3]

print("Using condition:")
print(f"iota: {iota_condition}, aspect ratio: {aspect_ratio_condition}, nfp: {nfp_condition}, helicity: {helicity_condition}")

# indices of the conditions
iota_idx = config.conditions.index('mean_iota')
nfp_idx = config.conditions.index('nfp')
helicity_idx = config.conditions.index('helicity')   
aspect_idx = config.conditions.index('aspect_ratio') 


# prepare conditions
conditions_raw = torch.ones((n_samples, len(config.conditions)), dtype=torch.float32)
conditions_raw[:, iota_idx] = iota_condition  # first column is iota
conditions_raw[:, nfp_idx] = nfp_condition  # second column is nfp
conditions_raw[:, helicity_idx] = helicity_condition  # third column is helicity
conditions_raw[:, aspect_idx] = aspect_ratio_condition  # last column is aspect ratio


# map back down to transformed space for evaluation
conditions_local, _, _ = to_standard(conditions_raw, Y_mean, Y_std)
conditions_local = conditions_local.float()


# sample
X_local = diffusion.sample(conditions_local)
X_local = X_local.cpu().detach().numpy()

# un-standardize samples for evaluation
X_raw = from_standard(X_local, X_mean, X_std)
if config.use_pca:
    X_raw = pca.inverse_transform(X_raw)

""" Evaluate the sampled configuration """


# storage
data = {
    'sqrt_qs_error_boozer': np.zeros(n_samples),
    'sqrt_qs_error_2term': np.zeros(n_samples),
    'sqrt_non_qs_error': np.zeros(n_samples),
    'aspect_ratio': np.zeros(n_samples),
    'iota_edge': np.zeros(n_samples),
    'mean_iota': np.zeros(n_samples),
    'success': np.zeros(n_samples, dtype=bool),
    'mean_iota_condition': conditions_raw[:, iota_idx].detach().numpy(),
    'aspect_ratio_condition': conditions_raw[:, aspect_idx].detach().numpy(),
    'nfp': np.round(conditions_raw[:, nfp_idx]).detach().numpy().astype(int), 
    'helicity': np.round(conditions_raw[:, helicity_idx]).detach().numpy().astype(int),
}

for ii, xx in enumerate(X_raw):
    print("")
    print(f"config {ii}/{n_samples}")

    # evaluate the configuration
    try:
        metrics, _ = evaluate_configuration_vmec(xx, round(nfp_condition), helicity=round(helicity_condition), vmec_input="../../diffusion_for_fusion/input.nfp4_template")
    except:
        print("Error evaluating configuration, skipping...")
        metrics = {
            'sqrt_qs_error_boozer': np.nan,
            'sqrt_qs_error_2term': np.nan,
            'sqrt_non_qs_error': np.nan,
            'aspect_ratio': np.nan,
            'iota_edge': np.nan,
            'mean_iota': np.nan,
            'success': False
        }

    # collect the data
    data['sqrt_qs_error_boozer'][ii] = metrics['sqrt_qs_error_boozer']
    data['sqrt_qs_error_2term'][ii] = metrics['sqrt_qs_error_2term']
    data['sqrt_non_qs_error'][ii] = metrics['sqrt_non_qs_error']
    data['iota_edge'][ii] = metrics['iota_edge']
    data['mean_iota'][ii] = metrics['mean_iota']
    data['aspect_ratio'][ii] = metrics['aspect_ratio']
    data['success'][ii] = metrics['success']
    print(f"success={metrics['success']}, sqrt_non_qs_error={metrics['sqrt_non_qs_error']}, aspect_ratio={metrics['aspect_ratio']}, mean_iota={metrics['mean_iota']}")


# save data
outdir = "./output/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
# save samples
outfilename = outdir + f'diffusion_samples_iota_{iota_condition}_nfp_{nfp_condition}_helicity_{helicity_condition}_aspect_ratio_{aspect_ratio_condition}.npy'
if os.path.exists(outfilename):
    existing_samples = np.load(outfilename)
    X_samples = np.concatenate([existing_samples, X_raw], axis=0)
else:
    X_samples = X_raw
print(len(X_samples))
np.save(outfilename, X_samples)
print(f"Samples saved to {outfilename}")

# save metrics
outfilename = outdir + f'diffusion_metrics_iota_{iota_condition}_nfp_{nfp_condition}_helicity_{helicity_condition}_aspect_ratio_{aspect_ratio_condition}.csv'
# append to existing file if it exists
if os.path.exists(outfilename):
    existing_df = pd.read_csv(outfilename)
    df = pd.concat([existing_df, pd.DataFrame(data)], ignore_index=True)
else:
    df = pd.DataFrame(data)
df.to_csv(outfilename, index=False)
print(df.head())
print(f"Metrics saved to {outfilename}")
