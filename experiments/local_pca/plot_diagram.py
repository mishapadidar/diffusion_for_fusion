import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from experiments.basic_conditional.load_quasr_data import load_quasr_data
from scipy.stats import gaussian_kde


""" 
Make a diagram to show how local PCA works.
"""

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 11})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]



# Rosenbrock
a = 1
b = 3
f = lambda x, y: (a - x)**2 + b*(y - x**2)**2
xopt = np.array([a, a**2])
lb = -1
ub = 2


# generate a dataset
X_train = np.random.uniform(low=lb, high=ub, size=(2000, 2))
fX_train = f(X_train[:, 0], X_train[:, 1])
f_cutoff = a/2
idx_keep = fX_train < f_cutoff
X_train = X_train[idx_keep]
fX_train = fX_train[idx_keep]


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# contour plot of the objective
x = np.linspace(lb, ub, 100)
y = np.linspace(lb, ub, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
levels = np.linspace(0, np.max(Z), 15)
levels = np.sort(np.concatenate(([a/4, a, b], levels)))
axes[0].contour(X, Y, Z, levels=levels, alpha=0.5, colors='grey')
axes[1].contour(X, Y, Z, levels=levels, alpha=0.5, colors='grey')

# plot the samples
axes[0].scatter(X_train[:, 0], X_train[:, 1], color='tab:blue', alpha=0.7)

# plot PCA directions
pca = PCA(n_components=2)
pca.fit(X_train)
components = pca.components_
mean = pca.mean_
t = np.linspace(-8, 2, 100)
linestyles = ['dashed', 'dotted']
# for ii in range(2):
#     direction = components[ii]
#     axes[0].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2, alpha=1.0,
#                  ls=linestyles[ii])
#     axes[1].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2, alpha=1.0,
#                  ls=linestyles[ii])
direction = components[0]
axes[0].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2, alpha=1.0,
                ls=linestyles[0])
axes[1].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2, alpha=1.0,
                ls=linestyles[0])

# plot local PCA data
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)
X_train_pca = pca.inverse_transform(X_train_pca)
axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='tab:orange', alpha=0.5)

for ii in range(2):
    axes[ii].set_xlim(lb,ub)
    axes[ii].set_ylim(lb,ub)
    # axes[ii].legend(loc='lower right')
    axes[ii].set_xticks([])
    axes[ii].set_yticks([])

axes[0].set_title("Samples")
axes[1].set_title("Projected Samples")


# generate more data for the histogram
X_train = np.random.uniform(low=lb, high=ub, size=(50000, 2))
fX_train = f(X_train[:, 0], X_train[:, 1])
idx_keep = fX_train < f_cutoff
X_train = X_train[idx_keep]
fX_train = fX_train[idx_keep]
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)
X_train_pca = pca.inverse_transform(X_train_pca)
fX_train_pca = f(X_train_pca[:, 0], X_train_pca[:, 1])

# Compute and plot the CDF of the objective value
sorted_fX_train = np.sort(fX_train)
cdf_pca = np.arange(1, len(sorted_fX_train) + 1) / len(sorted_fX_train)
axes[2].plot(sorted_fX_train, cdf_pca, color='tab:blue', lw=3, linestyle='dashed', label='Samples')

sorted_fX_train_pca = np.sort(fX_train_pca)
cdf_pca = np.arange(1, len(sorted_fX_train_pca) + 1) / len(sorted_fX_train_pca)
axes[2].plot(sorted_fX_train_pca, cdf_pca, color='tab:orange', lw=3, linestyle='dashdot', label='Projected Samples')

axes[2].grid(True, which='both', linestyle='-', linewidth=1, alpha=0.4, color='gray', zorder=0)
axes[2].set_title("Objective Value CDF")
# axes[2].set_xlabel("Objective Value")
# axes[2].set_ylabel("Probability")
axes[2].legend(loc='lower right')


plt.tight_layout()
fig.savefig("local_pca_diagram.pdf", format="pdf", bbox_inches="tight")
plt.show()
