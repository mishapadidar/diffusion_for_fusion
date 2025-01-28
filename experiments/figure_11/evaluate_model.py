import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from diffusion_for_fusion.ddpm_fusion import (init_diffusion_model_from_config, to_standard, from_standard)
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from load_quasr_data import load_quasr_data, plot_pca_data

# full dimension data
indir = "./output/fig9_300_hidden=1024_layer=5_schedule=linear_epoch=10000/run_uuid_7d6ee66c-d8e9-4aaa-a71b-0876ca1ebd94"

config_pickle = indir+"/config.pickle"
model_path = indir+"/model.pth"

# load the data
data = pickle.load(open(config_pickle, "rb"))
config = data['config']
input_dim = data['input_dim']
state_dict = torch.load(model_path)

# init the model
diffusion, model = init_diffusion_model_from_config(config, input_dim)
diffusion.load_state_dict(state_dict)
diffusion.eval()  # Set to evaluation mode
model.eval()  # Set to evaluation mode

# load the dataset
X, pca = load_quasr_data(return_pca=config.return_pca, fig=config.dataset)
print(np.shape(X))

# sample some stellarators
n_samples = 128
config.eval_batch_size = n_samples
samples = diffusion.sample(config.eval_batch_size, input_dim)
samples = samples.cpu().detach().numpy()

# un-standardize samples
_, mean, std = to_standard(X)
samples = from_standard(samples, mean, std)

# convert to full size x
if config.return_pca:
    samples = pca.inverse_transform(samples)
    X = pca.inverse_transform(X)


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
    print(f"{ii})", res)

outdata = {}
outdata['Y_samples'] = Y # evaluations
outdata['X_samples'] = samples # raw samples


# evaluate the PCA of the samples
if not config.return_pca:

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
        print(f"{ii})", res)
    
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
    print(f"{ii})", res)

# save the data
outdata = {}
outdata['Y'] = Y # evaluations
outdata['X'] = X # raw data
outfilename = indir + "/evaluations_actual.pickle"
pickle.dump(outdata, open(outfilename, "wb"))
print("dumped data to", outfilename)
