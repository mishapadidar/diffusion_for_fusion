import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from load_quasr_data import prepare_data_from_config

# full dimension data
indir = "./output/mean_iota/run_uuid_520aaf36-ad18-4bb8-a2ba-dd662b8498f7"

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
X_train, X_mean, X_std, Y_train, Y_mean, Y_std, pca = prepare_data_from_config(config)# print(np.shape(X))
print(X_train.shape)

# sample some stellarators
n_samples = 128
config.eval_batch_size = n_samples
cond_eval = generate_conditions_for_eval(Y_train, batch_size = config.eval_batch_size, from_train=True, seed=config.seed, as_tensor = True)

samples =diffusion.sample(cond_eval)
samples = samples.cpu().detach().numpy()

# un-standardize samples for VMEC
samples = from_standard(samples, X_mean, X_std)

# convert to full size x for VMEC
if config.use_pca:
    samples = pca.inverse_transform(samples)
    X = pca.inverse_transform(X_train)

# destandardize cond_eval for comparison
cond_eval = from_standard(cond_eval, Y_mean, Y_std)


""" Evaluate the sampled configuration """

# evaluate the samples
Y = np.zeros((n_samples, 3))
for ii, xx in enumerate(samples):
    res = evaluate_configuration(x=xx,
                        nfp=4,
                        mpol=10,
                        ntor=10,
                        helicity_n=1,
                        vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                        # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                        plot=False)
    Y[ii] = res
    print(f"{ii})", res, cond_eval[ii].numpy())

outdata = {}
outdata['Y_samples'] = Y # evaluations
outdata['X_samples'] = samples # raw samples


""" Project the samples onto the PCA plane then evaluate."""
if not config.use_pca:

    # project the samples onto the PCA plane
    samples = pca.transform(samples)
    samples = pca.inverse_transform(samples)

    Y = np.zeros((n_samples, 3))

    for ii, xx in enumerate(samples):
        res = evaluate_configuration(x=xx,
                            nfp=4,
                            mpol=10,
                            ntor=10,
                            helicity_n=1,
                            vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                            # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                            plot=False)
        Y[ii] = res
        print(f"{ii})", res, cond_eval[ii].numpy())

    
    outdata['Y_pca'] = Y # evaluations of projected samples
    outdata['X_pca'] = samples # projected samples

# save the data
outfilename = indir + "/evaluations_samples.pickle"
pickle.dump(outdata, open(outfilename, "wb"))
print("dumped data to", outfilename)


""" Evaluate the actual dataset """
idx = np.random.randint(0, len(X), n_samples)
X = X[idx]
# storage
Y = np.zeros((n_samples, 3))

# evaluate the actual data
for ii, xx in enumerate(X):
    res = evaluate_configuration(x=xx,
                        nfp=4,
                        mpol=10,
                        ntor=10,
                        helicity_n=1,
                        vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                        # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                        plot=False)
    Y[ii] = res
    print(f"{ii})", res, cond_eval[ii].numpy())

# save the data
outdata = {}
outdata['Y'] = Y # evaluations
outdata['X'] = X # raw data
outfilename = indir + "/evaluations_actual.pickle"
pickle.dump(outdata, open(outfilename, "wb"))
print("dumped data to", outfilename)
