import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard, to_standard
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from load_quasr_data import prepare_data_from_config

# # conditioned on (iota, aspect) for nfp=4, helicity=1;  trained on PCA-9
# indir = "./output/mean_iota_aspect_ratio/run_uuid_4123533c-960a-411b-9ff0-2a990e3eb305"
# # conditioned on (iota, aspect, nfp, helicity); trained on PCA-9
# indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_da5a3230-deca-4709-a775-76b7365fbbd2"
# # conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
# indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522"
# conditioned on (iota, aspect, nfp, helicity); trained on full data
# indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_6d0b93a0-238c-41bf-b794-5b6a467a2797"
# # conditioned on (iota, aspect, nfp, helicity); trained on full data with bigger model
# indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_c4793466-4f56-4860-b5d8-37d8aaef81f8"
# # conditioned on (iota, aspect, nfp, helicity); trained on PCA-200 w/ big model
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_1844e921-bfbe-4de5-a002-4ab92a213e7c"

# conditions; fig9 uses (1.1, 12, 4, 1)
n_samples = 256
mean_iota = 1.1
aspect_ratio = 12.0
nfp = 4
helicity = 1
use_local_pca = True
n_local_pca = 3

config_pickle = indir+"/config.pickle"
model_path = indir+"/model.pth"

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
print(X_train.shape)

# subset the data
Y_init = from_standard(Y_train, Y_mean, Y_std)
idx_subset = ((np.abs(Y_init[:,0] - mean_iota)/mean_iota < 0.01) & (np.abs(Y_init[:,1] - aspect_ratio)/aspect_ratio < 0.01) & 
              (Y_init[:,2] == 4) & (Y_init[:,3] == 1))
X_train = X_train[idx_subset]
Y_train = Y_train[idx_subset]

# X_train may be empty now
n_train = len(X_train)

# sample stellarators randomly
# cond_eval = generate_conditions_for_eval(Y_train, batch_size = n_samples, from_train=False, seed=config.seed, as_tensor = True)
# # make sure the integer features actually map to integers
# cond_eval_raw = from_standard(cond_eval, Y_mean, Y_std)
# cond_eval_raw[:,2:] = torch.round(cond_eval_raw[:,2:])
# cond_eval, _, _ = to_standard(cond_eval_raw, Y_mean, Y_std)
# cond_eval = cond_eval.type(torch.float32)

# use only one condition
cond_eval_raw = torch.zeros((n_samples, 4))
cond_eval_raw[:,0] = mean_iota # iota
cond_eval_raw[:,1] = aspect_ratio # aspect
cond_eval_raw[:,2] = nfp # nfp
cond_eval_raw[:,3] = helicity # helicity
cond_eval, _, _ = to_standard(cond_eval_raw, Y_mean, Y_std)
cond_eval = cond_eval.type(torch.float32)

# sample
samples = diffusion.sample(cond_eval)
samples = samples.cpu().detach().numpy()

# un-standardize samples for VMEC
samples = from_standard(samples, X_mean, X_std)
if n_train > 0:
    X_train = from_standard(X_train, X_mean, X_std)

# convert to full size x for VMEC
if config.use_pca:
    samples = pca.inverse_transform(samples)
    if n_train > 0:
        X_train = pca.inverse_transform(X_train)

# local PCA
if use_local_pca:
    pca = PCA(n_components=n_local_pca, svd_solver='full')
    pca.fit_transform(samples)
    samples = pca.transform(samples)
    samples = pca.inverse_transform(samples)

""" Plot the sampled configurations """

# PCA for plotting in 2D data
pca_plot = PCA(n_components=2, svd_solver='full')
samples_2d = pca_plot.fit_transform(samples)
if n_train > 0:
    X_2d = pca_plot.transform(X_train)
    plt.scatter(X_2d[:,0], X_2d[:,1], alpha=0.6, label='actual')
plt.scatter(samples_2d[:,0], samples_2d[:,1], alpha=0.6, label='diffusion')
plt.legend(loc='upper right')
plt.show()

""" Evaluate the sampled configuration """

# destandardize cond_eval for print statements and saving
cond_eval = from_standard(cond_eval, Y_mean, Y_std)

# evaluate the samples
Y = np.zeros((n_samples, 3))
for ii, xx in enumerate(samples):
    res = evaluate_configuration(x=xx,
                        nfp=nfp,
                        mpol=10,
                        ntor=10,
                        helicity_n=helicity,
                        # vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                        vmec_input="../../vmec_input_files/input.nfp4_torus",
                        # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                        plot=False)
    Y[ii] = res
    print(f"{ii})", res, cond_eval[ii].numpy())

# save the data
outfilename = indir + f"/evaluations_samples_iota_{mean_iota}_aspect_{aspect_ratio}_nfp_{nfp}_helicity_{helicity}_local_pca_{use_local_pca}_{n_local_pca}.pickle"
outdata = {}
outdata['Y_samples'] = Y # evaluations
outdata['X_samples'] = samples # raw samples
outdata['Y_conditions'] = cond_eval # target values of Y
outdata['mean_iota'] = mean_iota
outdata['aspect_ratio'] = aspect_ratio
outdata['nfp'] = nfp
outdata['helicity'] = helicity
outdata['n_local_pca'] = n_local_pca
outdata['use_local_pca'] = use_local_pca
pickle.dump(outdata, open(outfilename, "wb"))
print("dumped data to", outfilename)


""" Evaluate the actual dataset """
if n_train > 0:
    # random subset
    idx = np.random.randint(0, len(X_train), n_samples)
    X = X_train[idx]
    # storage
    Y = np.zeros((n_samples, 3))

    # evaluate the actual data
    for ii, xx in enumerate(X):
        res = evaluate_configuration(x=xx,
                            nfp=nfp,
                            mpol=10,
                            ntor=10,
                            helicity_n=helicity,
                            # vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                            vmec_input="../../vmec_input_files/input.nfp4_torus",
                            # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                            plot=False)
        Y[ii] = res
        print(f"{ii})", res)

    # save the data
    outdata = {}
    outdata['Y'] = Y # evaluations
    outdata['X'] = X # raw data
    outdata['mean_iota'] = mean_iota
    outdata['aspect_ratio'] = aspect_ratio
    outdata['nfp'] = nfp
    outdata['helicity'] = helicity
    outdata['n_local_pca'] = n_local_pca
    outfilename = indir + f"/evaluations_actual_iota_{mean_iota}_aspect_{aspect_ratio}_nfp_{nfp}_helicity_{helicity}_local_pca_{use_local_pca}_{n_local_pca}.pickle"
    pickle.dump(outdata, open(outfilename, "wb"))
    print("dumped data to", outfilename)
