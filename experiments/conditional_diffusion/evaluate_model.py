import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard, to_standard
# from diffusion_for_fusion.evaluate_configuration_sheet_curent import evaluate_configuration as evaluate_configuration_sheet
from diffusion_for_fusion.evaluate_configuration_vmec import evaluate_configuration as evaluate_configuration_vmec
from load_quasr_data import prepare_data_from_config
import os
import time

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"

# sample parameters
n_samples = 12
n_local_pca = 661


# turn on/off local PCA
use_local_pca = n_local_pca < 661

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
Y_raw = from_standard(Y_train, Y_mean, Y_std)
X_raw = from_standard(X_train, X_mean, X_std)
X_raw = pca.inverse_transform(X_raw)
n_dim_raw = np.shape(X_raw)[1]  # number of dimensions in the raw data

# get random condition
seed = int(time.time()) % (2**32 - 1)  # Use current time to generate a seed
cond_samples = generate_conditions_for_eval(Y_train, batch_size = n_samples, from_train=True, seed=seed, as_tensor = True)

# map conditions to the raw data space
cond_samples_raw = from_standard(cond_samples, Y_mean, Y_std)


# # uncomment to use figure 11 conditions
# print(config.conditions)
# idx = ((np.abs(Y_raw[:, 0] - 2.30)/2.30 < 0.001) & (np.abs(Y_raw[:, 1] - 12)/12 < 0.1)
#     & (Y_raw[:, 2] == 4) & (Y_raw[:, 3] == 1))
# X_train = X_train[idx]
# Y_train = Y_train[idx]
# X_raw = X_raw[idx]
# Y_raw = Y_raw[idx]
# cond_samples = torch.tensor(Y_train).type(torch.float32)

""" Evaluate the sampled configuration """

# indices of the conditions
iota_idx = config.conditions.index('mean_iota')
nfp_idx = config.conditions.index('nfp')
helicity_idx = config.conditions.index('helicity')   
aspect_idx = config.conditions.index('aspect_ratio') 

print("iota index:", iota_idx)
print("nfp index:", nfp_idx)
print("helicity index:", helicity_idx)
print("aspect ratio index:", aspect_idx)

# storage
data = {
    'sqrt_qs_error_boozer': np.zeros(n_samples),
    'sqrt_qs_error_2term': np.zeros(n_samples),
    'sqrt_non_qs_error': np.zeros(n_samples),
    'aspect_ratio': np.zeros(n_samples),
    'iota': np.zeros(n_samples),
    'success': np.zeros(n_samples, dtype=bool),
    'n_local_pca': n_local_pca*np.ones(n_samples, dtype=int),
    'use_local_pca': use_local_pca*np.ones(n_samples, dtype=bool),
    'iota_condition': cond_samples_raw[:, iota_idx].detach().numpy(),
    'aspect_ratio_condition': cond_samples_raw[:, aspect_idx].detach().numpy(),
    'nfp_condition': np.round(cond_samples_raw[:, nfp_idx]).detach().numpy().astype(int), 
    'helicity_condition': np.round(cond_samples_raw[:, helicity_idx]).detach().numpy().astype(int),
}

# storage for samples
X_samples = np.zeros((n_samples, n_dim_raw))

if use_local_pca:
    batch_size = 1024  # number of samples for local PCA
else:
    batch_size = 1

for ii in range(n_samples):

    print("")
    print(f"Configuration {ii}/{n_samples})")

    # # generate one condition
    cond_local = torch.ones((batch_size, len(config.conditions)), dtype=torch.float32)  # condition for local PCA
    cond_local = cond_local * cond_samples[ii, :]

    # sample
    X_local = diffusion.sample(cond_local)
    X_local = X_local.cpu().detach().numpy()

    # un-standardize samples for evaluation
    X_local = from_standard(X_local, X_mean, X_std)
    cond_local = from_standard(cond_local, Y_mean, Y_std)
    if config.use_pca:
        X_local = pca.inverse_transform(X_local)

    # # plot data
    # import matplotlib.pyplot as plt
    # local_pca = PCA(n_components=2, svd_solver='full')
    # X_raw_pca = local_pca.fit_transform(X_raw)
    # plt.scatter(X_raw_pca[:, 0], X_raw_pca[:, 1], label='Raw Data')
    # X_plot = pca.fit_transform(X_local)
    # plt.scatter(X_plot[:, 0], X_plot[:, 1], label='Sampled Data', alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.show()

    # local PCA
    if use_local_pca:
        local_pca = PCA(n_components=n_local_pca, svd_solver='full')
        X_local = local_pca.fit_transform(X_local)
        X_local = local_pca.inverse_transform(X_local)

    # only keep one point for evaluation
    xx = X_local[0]
    
    # evaluate the configuration
    # iota = cond_local[0, iota_idx].item()  # first column is mean_iota
    nfp = round(cond_local[0, nfp_idx].item())  # third column is nfp
    helicity = round(cond_local[0, helicity_idx].item())  # last column is helicity
    # field topology doesnt depend on G
    # metrics, _ = evaluate_configuration_sheet(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=10, N=10, G=1.0, ntheta=31, nphi=31, extend_factor=0.1)
    metrics, _ = evaluate_configuration_vmec(xx, nfp, helicity=helicity, vmec_input="../../diffusion_for_fusion/input.nfp4_template")

    # collect the data
    data['sqrt_qs_error_boozer'][ii] = metrics['sqrt_qs_error_boozer']
    data['sqrt_qs_error_2term'][ii] = metrics['sqrt_qs_error_2term']
    data['sqrt_non_qs_error'][ii] = metrics['sqrt_non_qs_error']
    data['iota'][ii] = metrics['iota']
    data['aspect_ratio'][ii] = metrics['aspect_ratio']
    data['success'][ii] = metrics['success']
    X_samples[ii, :] = xx
    # print("Configuration", ii, cond_local[0])
    print(f"success={metrics['success']}, sqrt_qs_error_boozer={metrics['sqrt_qs_error_boozer']}, aspect_ratio={metrics['aspect_ratio']}, iota={metrics['iota']}, nfp={nfp}, helicity={helicity}")


# save data
outdir = indir + "evaluations/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
# save samples
outfilename = outdir + f'diffusion_samples_local_pca_{n_local_pca}'
# np.save(outfilename, X_samples)
if os.path.exists(outfilename + '.npy'):
    existing_samples = np.load(outfilename + '.npy')
    X_samples = np.concatenate([existing_samples, X_samples], axis=0)
np.save(outfilename, X_samples)
print(f"Samples saved to {outfilename}")

# save metrics
outfilename = outdir + f'diffusion_metrics_local_pca_{n_local_pca}.csv'
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
