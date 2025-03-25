from diffusion_for_fusion.autoencoder import init_autoencoder_from_config
import numpy as np
from diffusion_for_fusion.ddpm_fusion import to_standard, from_standard
from experiments.basic_conditional.load_quasr_data import load_quasr_data
import pickle
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# point to the model directory
# indir = "./output/run_50_uuid_c3c36876-518d-4b0c-9c7d-ff408a0f2b87"
indir = "./output/run_5_uuid_3cdf040b-4d83-4483-b751-b1dbc10d657d"
config_filename = indir + "/config.pickle"
model_filename = indir + "/model.pth"

# load model
data = pickle.load(open(config_filename,"rb"))
config = data['config']
input_dim = data['input_dim']
autoencoder = init_autoencoder_from_config(config, input_dim)
autoencoder.load_state_dict(torch.load(model_filename))
autoencoder.eval()

# load and standardize data
X_train, Y_train = load_quasr_data(['mean_iota','aspect_ratio','nfp','helicity'])
X_train, X_mean, X_std = to_standard(X_train)
input_dim = np.shape(X_train)[1]

# subset the data; fig9 uses (1.1, 12, 4, 1); fig13 uses (1.3, 12, 4, 1)
mean_iota = 1.1
aspect_ratio = 12.0
nfp = 4
helicity = 1
idx_subset = ((np.abs(Y_train[:,0] - mean_iota)/mean_iota < 0.01) & (np.abs(Y_train[:,1] - aspect_ratio)/aspect_ratio < 0.01) & 
              (Y_train[:,2] == 4) & (Y_train[:,3] == 1))
X_train = X_train[idx_subset]
Y_train = Y_train[idx_subset]

# encode the data
X_train_decoded = autoencoder(torch.tensor(X_train).type(torch.float32)).detach().numpy()
print(np.mean((X_train - X_train_decoded)**2))

# perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_decoded_pca = pca.transform(X_train_decoded)

# plot the PCA result
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, label='Data')
plt.scatter(X_train_decoded_pca[:, 0], X_train_decoded_pca[:, 1], alpha=0.5, label='Decoded Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Decoded Data')
plt.legend(loc='upper right')
plt.show()