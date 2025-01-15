import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from diffusion_for_fusion.ddpm_fusion import (init_diffusion_model_from_config)
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from load_quasr_data import figure_11_data


indir = "./output/fig11_200_hidden=256_layer=3_schedule=linear_epoch=10000/run_uuid_eb907f97-9f41-423b-b232-0db3250b16fa"
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

# sample some stellarators
samples = diffusion.sample(config.eval_batch_size, input_dim)
samples = samples.cpu().detach().numpy()


# convert to full size x
if config.return_pca:
    X_pca, pca = figure_11_data(return_pca=True, return_pca_components=True)

    # plot the samples in the PCA plane
    plt.scatter(X_pca[:,0], X_pca[:,1], label='actual')
    plt.scatter(samples[:,0], samples[:,1], label='diffusion')
    plt.legend(loc='upper right')
    plt.show()

    X = pca.inverse_transform(X_pca)
    samples = pca.inverse_transform(samples)

X= figure_11_data(return_pca=False)

# now call VMEC
for ii, xx in enumerate(X):
    res = evaluate_configuration(x=xx,
                        nfp=4,
                        mpol=10,
                        ntor=10,
                        helicity_n=1,
                        vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res",
                        # vmec_input="../../vmec_input_files/input.new_QH_andrew",
                        plot=False)
    print(res)