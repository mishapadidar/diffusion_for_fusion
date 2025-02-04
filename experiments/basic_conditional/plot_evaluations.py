import pickle
import matplotlib.pyplot as plt
import numpy as np
from load_quasr_data import prepare_data_from_config
from diffusion_for_fusion.ddpm_fusion import from_standard


plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 14})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

indir = "./output/mean_iota_aspect_ratio/run_uuid_4123533c-960a-411b-9ff0-2a990e3eb305"
sample_filename = indir + "/evaluations_samples.pickle"
data_filename = indir + "/evaluations_actual.pickle"
config_pickle = indir + "/config.pickle"

# load the config
with open(config_pickle, 'rb') as f:
    data = pickle.load(f)
    config = data['config']

# load, PCA, and standardize data
# X_train, X_mean, X_std, Y_train, Y_mean, Y_std, pca = prepare_data_from_config(config)

# samples from diffusion model
with open(sample_filename, 'rb') as f:
    data = pickle.load(f)
X_samples = data['X_samples']
Y_samples = data['Y_samples']
Y_pca = data.get('Y_pca', None)
Y_cond =  data['Y_cond'].numpy()

""" Plot fig9 samples """

# # actual evaluations
# with open(data_filename, 'rb') as f:
#     data = pickle.load(f)
# Y_actual = data['Y']


# drop eval failures
idx_keep = np.all(Y_samples !=0.0, axis=1)
Y = Y_samples[idx_keep]
if Y_pca:
    Y_pca = Y_pca[idx_keep]


plt.figure(figsize=(10, 6))
plt.hist(Y[:, 1], alpha=0.4, color=colors[0], density=True, label='Diffusion')
if Y_pca:
    plt.hist(Y_pca[:, 1], alpha=0.4, color=colors[1], density=True, label='PCA projection of diffusion samples')
plt.hist(Y_cond[:, 0], alpha=0.4, color=colors[2], density=True, label='target')
plt.xlabel("mean rotational transform")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(Y[:, 2], alpha=0.4, color=colors[0], density=True, label='Diffusion ')
if Y_pca:
    plt.hist(Y_pca[:, 2], alpha=0.4, color=colors[1], density=True, label='PCA projection of diffusion samples')
plt.hist(Y_cond[:, 1], alpha=0.4, color=colors[2], density=True, label='actual')
plt.xlabel("aspect ratio")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(Y[:, 0], alpha=0.4, color=colors[0], density=True, label='Diffusion ')
if Y_pca:
    plt.hist(Y_pca[:, 0], alpha=0.4, color=colors[1], density=True, label='PCA projection of diffusion samples')
plt.xlabel("Quasi-symmetry Error")
plt.legend()
plt.show()

# fig, (ax1,ax2)  = plt.subplots(1, 2, figsize=(10, 6))
# ax1.hist(Y[:, 0], alpha=0.4, color=colors[0], density=True, label='Diffusion')
# ax1.set_xlabel('Quasi-symmetry Error')
# ax1.legend()
# if Y_pca:
#     ax2.hist(Y_pca[:, 0], alpha=0.4, color=colors[1], density=True, label='PCA proj of diffusion')
# ax2.hist(Y_actual[:, 0], alpha=0.4, color=colors[2], density=True, label='actual')
# ax2.legend()
# ax2.set_xlabel('Quasi-symmetry Error')
# ax2.set_title("PCA projection of diffusion samples")
# plt.show()

