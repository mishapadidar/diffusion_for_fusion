import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 14})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

indir = "./output/fig9_300_hidden=1024_layer=5_schedule=linear_epoch=10000/run_uuid_7d6ee66c-d8e9-4aaa-a71b-0876ca1ebd94"
sample_filename = indir + "/evaluations_samples.pickle"
data_filename = indir + "/evaluations_actual.pickle"

# Load the pickle file
with open(sample_filename, 'rb') as f:
    data = pickle.load(f)
# Extract X and Y arrays from the dictionary
Y = data['Y']
Y_pca = data['Y_pca']

with open(data_filename, 'rb') as f:
    data = pickle.load(f)
Y_actual = data['Y']


# drop eval failures
idx_keep = np.all(Y !=0.0, axis=1)
Y = Y[idx_keep]
Y_pca = Y_pca[idx_keep]
idx_keep = np.all(Y_actual !=0.0, axis=1)
Y_actual = Y_actual[idx_keep]


plt.figure(figsize=(10, 6))
plt.hist(Y[:, 1], alpha=0.4, color=colors[0], density=True, label='Diffusion (all dim)')
plt.hist(Y_pca[:, 1], alpha=0.4, color=colors[1], density=True, label='PCA projection of diffusion samples')
# plt.hist(Y_actual[:, 1], alpha=0.4, color=colors[2], density=True, label='actual')
plt.xlabel("mean rotational transform")
plt.axvline(x=1.1, linestyle='--', color='k', lw=2)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(Y[:, 2], alpha=0.4, color=colors[0], density=True, label='Diffusion (all dim)')
plt.hist(Y_pca[:, 2], alpha=0.4, color=colors[1], density=True, label='PCA projection of diffusion samples')
# plt.hist(Y_actual[:, 2], alpha=0.4, color=colors[2], density=True, label='actual')
plt.xlabel("aspect ratio")
plt.axvline(x=12.0, linestyle='--', color='k', lw=2)
plt.legend()
plt.show()


fig, (ax1,ax2)  = plt.subplots(1, 2, figsize=(10, 6))
ax1.hist(Y[:, 0], alpha=0.4, color=colors[0], density=True, label='Diffusion (all dim)')
ax1.set_xlabel('Quasi-symmetry Error')
ax1.legend()
ax2.hist(Y_pca[:, 0], alpha=0.4, color=colors[1], density=True, label='PCA proj of diffusion')
ax2.hist(Y_actual[:, 0], alpha=0.4, color=colors[2], density=True, label='actual')
ax2.legend()
ax2.set_xlabel('Quasi-symmetry Error')
ax2.set_title("PCA projection of diffusion samples")
plt.show()

