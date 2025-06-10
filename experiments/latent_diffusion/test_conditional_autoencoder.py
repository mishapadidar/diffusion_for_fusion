from diffusion_for_fusion.autoencoder import init_conditional_autoencoder_from_config
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
import numpy as np
from diffusion_for_fusion.ddpm_fusion import to_standard, from_standard
from experiments.conditional_diffusion.load_quasr_data import load_quasr_data
import pickle
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# point to the model directory
# indir = "./output/run_5_uuid_c14c1262-4c0a-459b-b34a-8626d63d2c42"
# indir = "./output/run_50_uuid_6baec103-5abf-43e1-81c2-da4184baea7c"
# indir = "./output/run_100_uuid_745fc474-009d-4b9e-9bd3-9711631d8120" # pretty good!
indir = "./output/run_200_uuid_cd089466-40c4-4729-bb09-b272976d942c" # even better!

config_filename = indir + "/config.pickle"
model_filename = indir + "/model.pth"

# load model
data = pickle.load(open(config_filename,"rb"))
config = data['config']
input_dim = data['input_dim']
autoencoder = init_conditional_autoencoder_from_config(config, input_dim)
autoencoder.load_state_dict(torch.load(model_filename))
autoencoder.eval()

# load data
X_train, Y_train = load_quasr_data(config.conditions)

# global PCA
pca = PCA(n_components=config.encoding_size)
X_train_global_pca = pca.fit_transform(X_train)
X_train_global_pca = pca.inverse_transform(X_train_global_pca)

# find subset the data; fig9 uses (1.1, 12, 4, 1); fig13 uses (1.3, 12, 4, 1)
mean_iota = 1.1
aspect_ratio = 12.0
nfp = 4
helicity = 1
idx_subset = ((np.abs(Y_train[:,0] - mean_iota)/mean_iota < 0.01) & (np.abs(Y_train[:,1] - aspect_ratio)/aspect_ratio < 0.01) & 
              (Y_train[:,2] == 4) & (Y_train[:,3] == 1))

# standardize data
X_train, X_mean, X_std = to_standard(X_train)
Y_train, Y_mean, Y_std = to_standard(Y_train)
input_dim = np.shape(X_train)[1]

# subset the data
X_train = X_train[idx_subset]
Y_train = Y_train[idx_subset]
X_train_global_pca = X_train_global_pca[idx_subset]

# encode/decode the data
X_train_torch = torch.tensor(X_train).type(torch.float32)
Y_train_torch = torch.tensor(Y_train).type(torch.float32)

X_train_decoded = autoencoder(X_train_torch, Y_train_torch).detach().numpy()

# de-standardize data
X_train = from_standard(X_train, X_mean, X_std)
X_train_decoded = from_standard(X_train_decoded, X_mean, X_std)

# perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_decoded_pca = pca.transform(X_train_decoded)

# plot data on the 2D PCA plane
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, label='Data')
plt.scatter(X_train_decoded_pca[:, 0], X_train_decoded_pca[:, 1], alpha=0.5, label='Decoded Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Data')
plt.legend(loc='upper right')
plt.show()


""" Evaluate the decoded train data """

# sample the data
n_samples = 100

np.random.seed(0)

idx_samples = np.random.choice(np.arange(len(X_train)), size=n_samples, replace=False)
X_samples = X_train_decoded[idx_samples]
X_actuals = X_train[idx_samples]
# X_actuals = X_train_global_pca[idx_samples]
Y_targets = Y_train[idx_samples]

# evaluate the samples
Y_samples = np.zeros((n_samples, 3))
Y_actuals = np.zeros((n_samples, 3))
for ii in range(n_samples):
    # evaluate sample
    Y_samples[ii] = evaluate_configuration(x=X_samples[ii],
                        nfp=nfp,
                        mpol=10,
                        ntor=10,
                        helicity_n=helicity,
                        vmec_input="../../vmec_input_files/input.nfp4_torus",
                        plot=False)

    # evaluate actual
    Y_actuals[ii] = evaluate_configuration(x=X_actuals[ii],
                        nfp=nfp,
                        mpol=10,
                        ntor=10,
                        helicity_n=helicity,
                        vmec_input="../../vmec_input_files/input.nfp4_torus",
                        plot=False)
    print(f"{ii})", Y_samples[ii], Y_actuals[ii])

# save the data
outfilename = indir + f"/autoencoder_vmec_evals_iota_{mean_iota}_aspect_{aspect_ratio}_nfp_{nfp}_helicity_{helicity}.pickle"
outdata = {}
outdata['X_samples'] = X_samples # samples
outdata['Y_samples'] = Y_samples # evaluations
outdata['X_actuals'] = X_actuals # actuals
outdata['Y_actuals'] = Y_actuals # evaluations
outdata['mean_iota'] = mean_iota
outdata['aspect_ratio'] = aspect_ratio
outdata['nfp'] = nfp
outdata['helicity'] = helicity
pickle.dump(outdata, open(outfilename, "wb"))
print("dumped data to", outfilename)