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
import os, sys
import time

"""
Evaluate the training data used to train the conditional diffusion model.
This provides a performance baseline for the model -- it can't be better than the training data.

    python evaluate_baseline.py n_samples target_nfp

Args:
    n_samples (int): number of samples to generate
    target_nfp (int, str): either "all" or the integer nfp value. 
        Downselects the conditions to only those with the given nfp value.

"""

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "../conditional_diffusion/output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"

# sample parameters
n_samples = int(sys.argv[1])
target_nfp = sys.argv[2]

# sample only a subset of the data
if target_nfp == "all":
    fix_nfp = False
else:
    target_nfp = int(target_nfp)
    fix_nfp = True

print("Sampling n_samples =", n_samples, "with target_nfp =", target_nfp)

""" Load samples """

config_pickle = indir+"config.pickle"
model_path = indir+"model.pth"

# load the data
data = pickle.load(open(config_pickle, "rb"))
config = data['config']
state_dict = torch.load(model_path)

# load, PCA, and standardize data
X_train, X_mean, X_std, Y_train, Y_mean, Y_std, pca = prepare_data_from_config(config)
Y_raw = from_standard(Y_train, Y_mean, Y_std)
X_raw = from_standard(X_train, X_mean, X_std)
X_raw = pca.inverse_transform(X_raw)

# indices of the conditions
iota_idx = config.conditions.index('mean_iota')
aspect_idx = config.conditions.index('aspect_ratio')
nfp_idx = config.conditions.index('nfp')
helicity_idx = config.conditions.index('helicity')    

print("iota index:", iota_idx)
print("nfp index:", nfp_idx)
print("helicity index:", helicity_idx)

# only keep data with the target nfp
if fix_nfp:
    idx_keep = np.where(np.round(Y_raw[:, nfp_idx]).astype(int) == target_nfp)[0]
    print(f"Keeping {len(idx_keep)}/{len(Y_raw)} samples with nfp={target_nfp}.")
    X_raw = X_raw[idx_keep]
    Y_raw = Y_raw[idx_keep]

# downsample
idx_samples = np.random.choice(len(X_raw), n_samples, replace=False)
X_samples = X_raw[idx_samples]
Y_samples = Y_raw[idx_samples]


""" Evaluate the data """


# storage
data = {
    'sqrt_qs_error_boozer': np.zeros(n_samples),
    'sqrt_qs_error_2term': np.zeros(n_samples),
    'sqrt_non_qs_error': np.zeros(n_samples),
    'aspect_ratio': np.zeros(n_samples),
    'mean_iota': np.zeros(n_samples),
    'iota_edge': np.zeros(n_samples),
    'success': np.zeros(n_samples, dtype=bool),
    'nfp': np.round(Y_samples[:, nfp_idx]).astype(int), # from dataset
    'helicity': np.round(Y_samples[:, helicity_idx]).astype(int), # from dataset
    'mean_iota_condition': Y_samples[:, iota_idx].astype(float),  # from dataset
    'aspect_ratio_condition': Y_samples[:, aspect_idx].astype(float),  # from dataset
}


for ii, xx in enumerate(X_samples):

    print("")
    print(f"Configuration {ii}/{n_samples})")

    # evaluate the configuration
    mean_iota_condition = Y_samples[ii, iota_idx].item()  # first column is mean_iota
    aspect_condition = Y_samples[ii, aspect_idx].item()  # second column is aspect_ratio
    nfp = round(Y_samples[ii, nfp_idx].item())  # third column is nfp
    helicity = round(Y_samples[ii, helicity_idx].item())  # last column is helicity
    # field topology doesnt depend on G
    # metrics, _ = evaluate_configuration_sheet(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=10, N=10, G=1.0, ntheta=31, nphi=31, extend_factor=0.1)
    metrics, _ = evaluate_configuration_vmec(xx, nfp, helicity=helicity, vmec_input="../../diffusion_for_fusion/input.nfp4_template")

    # collect the data  
    data['sqrt_qs_error_boozer'][ii] = metrics['sqrt_qs_error_boozer']
    data['sqrt_qs_error_2term'][ii] = metrics['sqrt_qs_error_2term']
    data['sqrt_non_qs_error'][ii] = metrics['sqrt_non_qs_error']
    data['mean_iota'][ii] =  metrics['mean_iota']
    data['iota_edge'][ii] =  metrics['iota_edge']
    data['aspect_ratio'][ii] = metrics['aspect_ratio']
    data['success'][ii] = metrics['success']

    X_samples[ii, :] = xx
    print(f"Actuals: iota {mean_iota_condition}, aspect {aspect_condition} nfp={nfp}, helicity={helicity}.")
    print(f"Estimates: sqrt_qs_error_boozer={metrics['sqrt_qs_error_boozer']}, mean_iota={metrics['mean_iota']}, aspect_ratio={metrics['aspect_ratio']}, nfp={nfp}, helicity={helicity}")


# save data
outdir = "./output/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
# save samples
outfilename = outdir + f'baseline_samples_nfp_{target_nfp}'
# np.save(outfilename, X_samples)
if os.path.exists(outfilename + '.npy'):
    existing_samples = np.load(outfilename + '.npy')
    X_samples = np.concatenate([existing_samples, X_samples], axis=0)
np.save(outfilename, X_samples)
print(f"Samples saved to {outfilename}")

# save metrics
outfilename = outdir + f'baseline_metrics_nfp_{target_nfp}.csv'
# df = pd.DataFrame(data)
# append to existing file if it exists
if os.path.exists(outfilename):
    existing_df = pd.read_csv(outfilename)
    df = pd.concat([existing_df, pd.DataFrame(data)], ignore_index=True)
else:
    df = pd.DataFrame(data)
df.to_csv(outfilename, index=False)
print(df.head())
print(f"Metrics saved to {outfilename}")
