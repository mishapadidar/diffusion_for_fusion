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

# # conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"

# model with exponential scaling
# indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_4e5d467e-e6fb-43f3-98d2-331059802049"

# sample parameters
n_samples = 50
n_local_pca = 5


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
_, X_mean, X_std, Y_train, Y_mean, Y_std, pca = prepare_data_from_config(config)
# Y_raw = from_standard(Y_train, Y_mean, Y_std)

# sample stellarators randomly
batch_size = max(n_samples, 1000) # to avoid too small batch size for local pca
cond_eval = generate_conditions_for_eval(Y_train, batch_size = batch_size, from_train=True, seed=config.seed, as_tensor = True)
cond_eval = cond_eval.type(torch.float32)

# sample
X_samples = diffusion.sample(cond_eval)
X_samples = X_samples.cpu().detach().numpy()

# un-standardize samples for evaluation
X_samples = from_standard(X_samples, X_mean, X_std)
cond_eval = from_standard(cond_eval, Y_mean, Y_std)

# convert to full size x
if config.use_pca:
    X_samples = pca.inverse_transform(X_samples)

# local PCA
if use_local_pca:
    pca = PCA(n_components=n_local_pca, svd_solver='full')
    pca.fit_transform(X_samples)
    X_samples = pca.transform(X_samples)
    X_samples = pca.inverse_transform(X_samples)

# only keep first n_samples
X_samples = X_samples[:n_samples]
cond_eval = cond_eval[:n_samples]


""" Evaluate the sampled configuration """

# we have to change the save order otherwise
assert config.conditions[0] == 'mean_iota', "mean_iota should be the first condition"
assert config.conditions[1] == 'aspect_ratio', "aspect_ratio should be the second condition"
assert config.conditions[2] == 'nfp',   "nfp should be the third condition"
assert config.conditions[3] == 'helicity',  "helicity should be the last condition"
                                

# storage
data = {
    'sqrt_qs_error': np.zeros(n_samples),
    'iota': np.zeros(n_samples),
    'aspect_ratio': np.zeros(n_samples),
    'boozer_residual_mse': np.zeros(n_samples),
    # 'mean_iota_condition': cond_eval[:, 0].numpy(),  # mean_iota is the first condition
    # 'aspect_ratio_condition': cond_eval[:, 1].numpy(),  # aspect_ratio is the second condition
    # 'nfp_condition': cond_eval[:, 2].numpy(),  # nfp is the third condition
    # 'helicity_condition': cond_eval[:, 3].numpy(),  # helicity is the last condition
    'n_local_pca': n_local_pca*np.ones(n_samples, dtype=int),
    'use_local_pca': use_local_pca*np.ones(n_samples, dtype=bool)
}
for cond in config.conditions:
    data[cond + '_condition'] = cond_eval[:, config.conditions.index(cond)].numpy()


for ii, xx in enumerate(X_samples):

    print("")
    
    # evaluate the configuration
    iota = cond_eval[ii, 0].item()  # first column is mean_iota
    nfp = int(cond_eval[ii, 2].item())  # third column is nfp
    helicity = cond_eval[ii, 3].item()  # last column is helicity
    # field topology doesnt depend on G
    metrics, _ = evaluate_configuration_sheet(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=10, N=10, G=1.0, ntheta=31, nphi=31, extend_factor=0.1)

    # collect the data  
    data['sqrt_qs_error'][ii] = metrics['sqrt_qs_error']
    data['iota'][ii] = iota
    data['aspect_ratio'][ii] = metrics['aspect_ratio']
    data['boozer_residual_mse'][ii] = metrics['boozer_residual_mse']
    print(f"sqrt_qs_error={metrics['sqrt_qs_error']}, iota={iota}, aspect_ratio={metrics['aspect_ratio']}, nfp={nfp}, helicity={helicity}, boozer_residual_mse={metrics['boozer_residual_mse']}")


# save data
outdir = indir + "evaluations/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
# save samples
outfilename = outdir + f'samples_pca_{n_local_pca}.pkl'
np.save(outfilename, X_samples)
print(f"Samples saved to {outfilename}")
# save metrics
outfilename = outdir + f'metrics_pca_{n_local_pca}.pkl'
df = pd.DataFrame(data)
# pd.to_pickle(df, outfilename)
df.to_csv(outfilename, index=False)
print(df.head())
print(f"Metrics saved to {outfilename}")
