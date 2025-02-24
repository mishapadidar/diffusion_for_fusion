import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from diffusion_for_fusion.ddpm_conditional_diffusion import (init_conditional_diffusion_model_from_config, generate_conditions_for_eval)
from diffusion_for_fusion.ddpm_fusion import from_standard
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from load_quasr_data import prepare_data_from_config
from sklearn.decomposition import PCA



plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 14})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


# # conditioned on (iota, aspect) for nfp=4, helicity=1;  trained on PCA-9
# indir = "./output/mean_iota_aspect_ratio/run_uuid_4123533c-960a-411b-9ff0-2a990e3eb305"
# conditioned on (iota, aspect, nfp, helicity); trained on PCA-9
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_da5a3230-deca-4709-a775-76b7365fbbd2"

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
print(np.shape(X_train))

# subset some data
Y_init = from_standard(Y_train, Y_mean, Y_std)
# idx_subset = ((np.abs(Y_init[:,0] - 1.1)/1.1 < 0.001) & (np.abs(Y_init[:,1] - 12)/12 < 0.01)) # fig 9
# idx_subset = ((np.abs(Y_init[:,0]- 2.30)/2.30 < 0.001) & (np.abs(Y_init[:,1] - 12)/12 < 0.1)) # fig 11
# idx_subset = ((np.abs(Y_init[:,0] - 1.1)/1.1 < 0.001) & (np.abs(Y_init[:,1] - 12)/12 < 0.01) & 
#               (Y_init[:,2] == 4) & (Y_init[:,3] == 1)) # fig 9
idx_subset = ((np.abs(Y_init[:,0]- 2.30)/2.30 < 0.001) & (np.abs(Y_init[:,1] - 12)/12 < 0.1) & 
              (Y_init[:,2] == 4) & (Y_init[:,3] == 1)) # fig 11

X_train = X_train[idx_subset]
Y_train = Y_train[idx_subset]
print(np.shape(X_train))


# generate conditions from data
n_samples = 256
cond_eval = generate_conditions_for_eval(Y_train, batch_size = n_samples, from_train=True, as_tensor = True)

# sample from the diffusion model
samples =diffusion.sample(cond_eval)
samples = samples.cpu().detach().numpy()


# plot the location of the samples in Y-space
Y_train = from_standard(Y_train, Y_mean, Y_std)
cond_eval = from_standard(cond_eval, Y_mean, Y_std)
plt.scatter(Y_train[:,0], Y_train[:,1], alpha=0.6, label='actual')
plt.scatter(cond_eval[:,0], cond_eval[:,1], alpha=0.6, label='diffusion')
plt.xlabel('mean rotational transform')
plt.ylabel('aspect ratio')
plt.legend(loc='upper right')
plt.show()

# PCA for plotting in 2D data
pca_plot = PCA(n_components=2, svd_solver='full')
X_2d = pca_plot.fit_transform(X_train)
samples_2d = pca_plot.transform(samples)

plt.scatter(X_2d[:,0], X_2d[:,1], alpha=0.6, label='actual')
plt.scatter(samples_2d[:,0], samples_2d[:,1], alpha=0.6, label='diffusion')
plt.legend(loc='upper right')
plt.show()



