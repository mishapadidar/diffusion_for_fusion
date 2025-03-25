from diffusion_for_fusion.autoencoder import init_autoencoder_from_config
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
import numpy as np
from diffusion_for_fusion.ddpm_fusion import to_standard, from_standard
from experiments.basic_conditional.load_quasr_data import load_quasr_data
import pickle
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# point to the model directory
# indir = "./output/run_5_uuid_3cdf040b-4d83-4483-b751-b1dbc10d657d"
# indir = "./output/run_50_uuid_c3c36876-518d-4b0c-9c7d-ff408a0f2b87"
# indir = "./output/run_100_uuid_2ad2d52d-a4cc-4ff7-b8ca-ab6f0728683a"
indir = "./output/run_100_uuid_5dadd407-f11e-4524-aaff-3e56bd9b34d6"

# trained on resnet
# indir = "./output/run_50_uuid_1def09e0-4e64-4d36-acd5-dfcf334a200d"
# indir = "./output/run_5_uuid_1c960b3c-5eac-41d5-92ce-6add5a103c77"

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


# encode/decode the data
X_train_torch = torch.tensor(X_train).type(torch.float32)
X_train_decoded = autoencoder(X_train_torch).detach().numpy()
# X_train_encoded = (autoencoder.encoder(X_train_torch) + autoencoder.encoder_residual(X_train_torch)).detach().numpy()
X_train_encoded = (autoencoder.encode(X_train_torch)).detach().numpy()

# test set
std = np.std(X_train, axis=0)
X_test = X_train + std * np.random.randn(*np.shape(X_train))
X_test_decoded = autoencoder(torch.tensor(X_test).type(torch.float32)).detach().numpy()
X_test_encoded = (autoencoder.encode(torch.tensor(X_test).type(torch.float32))).detach().numpy()

# de-standardize data
X_train = from_standard(X_train, X_mean, X_std)
X_train_decoded = from_standard(X_train_decoded, X_mean, X_std)
X_test_decoded = from_standard(X_test_decoded, X_mean, X_std)

# perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_decoded_pca = pca.transform(X_train_decoded)
X_test_decoded_pca = pca.transform(X_test_decoded)

# plot data on the 2D PCA plane
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, label='Data')
plt.scatter(X_train_decoded_pca[:, 0], X_train_decoded_pca[:, 1], alpha=0.5, label='Decoded Data')
plt.scatter(X_test_decoded_pca[:, 0], X_test_decoded_pca[:, 1], alpha=0.2, label='Decoded Test Data')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Data')
plt.legend(loc='upper right')
plt.show()

# plot encoded data on the 2D PCA plane
X_train_encoded_pca = pca.fit_transform(X_train_encoded)
X_test_encoded_pca = pca.transform(X_test_encoded)
plt.scatter(X_train_encoded_pca[:, 0], X_train_encoded_pca[:, 1], alpha=0.5, label='Encoded Data')
plt.scatter(X_test_encoded_pca[:, 0], X_test_encoded_pca[:, 1], alpha=0.2, label='Encoded Test Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Encoded Data')
plt.legend(loc='upper right')
plt.show()


""" Evaluate the decoded train data """

# sample the data
n_samples = 100

idx_samples = np.random.choice(np.arange(len(X_train)), size=n_samples, replace=False)
X_samples = X_train_decoded[idx_samples]
X_actuals = X_train[idx_samples]
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