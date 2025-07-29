import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard, to_standard
from diffusion_for_fusion.evaluate_configuration_sheet_curent import evaluate_configuration as evaluate_configuration_sheet
from load_quasr_data import prepare_data_from_config
import os
import time

"""
Evaluate the training data used to train the conditional diffusion model.
This provides a performance baseline for the model -- it can't be better than the training data.
"""

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"

# sample parameters
n_samples = 32

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

# # uncomment to use figure 11 conditions
# print(config.conditions)
# idx = ((np.abs(Y_raw[:, 0] - 2.30)/2.30 < 0.001) & (np.abs(Y_raw[:, 1] - 12)/12 < 0.1)
#     & (Y_raw[:, 2] == 4) & (Y_raw[:, 3] == 1))
# X_train = X_train[idx]
# Y_train = Y_train[idx]
# X_raw = X_raw[idx]
# Y_raw = Y_raw[idx]
# cond_samples = torch.tensor(Y_train).type(torch.float32)

# # plot data
# import matplotlib.pyplot as plt
# local_pca = PCA(n_components=2, svd_solver='full')
# X_raw_pca = local_pca.fit_transform(X_raw)
# plt.scatter(X_raw_pca[:, 0], X_raw_pca[:, 1], label='Raw Data')
# X_plot = pca.fit_transform(X_local)
# plt.scatter(X_plot[:, 0], X_plot[:, 1], label='Sampled Data', alpha=0.5)
# plt.legend(loc='upper right')
# plt.show()

# downsample
idx_samples = np.random.choice(len(X_raw), n_samples, replace=False)
X_samples = X_raw[idx_samples]
Y_samples = Y_raw[idx_samples]

# TODO: remove
print(Y_samples)

# indices of the conditions
iota_idx = config.conditions.index('mean_iota')
aspect_idx = config.conditions.index('aspect_ratio')
nfp_idx = config.conditions.index('nfp')
helicity_idx = config.conditions.index('helicity')    

print("iota index:", iota_idx)
print("nfp index:", nfp_idx)
print("helicity index:", helicity_idx)


""" Evaluate the data """


# storage
data = {
    'sqrt_qs_error': np.zeros(n_samples),
    'iota': np.zeros(n_samples),
    'aspect_ratio': np.zeros(n_samples),
    'boozer_residual_mse': np.zeros(n_samples),
}


for ii, xx in enumerate(X_samples):

    print("")
    print(f"Configuration {ii}/{n_samples})")

    # evaluate the configuration
    iota = Y_samples[ii, iota_idx].item()  # first column is mean_iota
    nfp = round(Y_samples[ii, nfp_idx].item())  # third column is nfp
    helicity = round(Y_samples[ii, helicity_idx].item())  # last column is helicity
    # field topology doesnt depend on G
    metrics, _ = evaluate_configuration_sheet(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=10, N=10, G=1.0, ntheta=31, nphi=31, extend_factor=0.1)

    # collect the data  
    data['sqrt_qs_error'][ii] = metrics['sqrt_qs_error']
    data['iota'][ii] = iota
    data['aspect_ratio'][ii] = metrics['aspect_ratio']
    data['boozer_residual_mse'][ii] = metrics['boozer_residual_mse']
    X_samples[ii, :] = xx
    print(f"Actuals: iota {iota}, aspect {Y_samples[ii, aspect_idx].item()} nfp={nfp}, helicity={helicity}.")
    print(f"Estimates: sqrt_qs_error={metrics['sqrt_qs_error']}, iota={iota}, aspect_ratio={metrics['aspect_ratio']}, nfp={nfp}, helicity={helicity}, boozer_residual_mse={metrics['boozer_residual_mse']}")


# save data
outdir = indir + "evaluations/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
# save samples
outfilename = outdir + f'baseline_samples'
# np.save(outfilename, X_samples)
if os.path.exists(outfilename + '.npy'):
    existing_samples = np.load(outfilename + '.npy')
    X_samples = np.concatenate([existing_samples, X_samples], axis=0)
np.save(outfilename, X_samples)
print(f"Samples saved to {outfilename}")

# save metrics
outfilename = outdir + f'baseline_metrics.csv'
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
