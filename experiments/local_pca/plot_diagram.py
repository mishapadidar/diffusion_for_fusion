import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from experiments.basic_conditional.load_quasr_data import load_quasr_data


""" 
Make a diagram to show how local PCA works.
"""

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 11})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]



# Rosenbrock
a = 1
b = 3
f = lambda x, y: (a - x)**2 + b*(y - x**2)**2
xopt = np.array([a, a**2])
lb = -1
ub = 2

X_train = np.random.uniform(low=lb, high=ub, size=(1000, 2))
fX_train = f(X_train[:, 0], X_train[:, 1])
idx_keep = fX_train < a
X_train = X_train[idx_keep]
fX_train = fX_train[idx_keep]


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
x = np.linspace(lb, ub, 100)
y = np.linspace(lb, ub, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
levels = np.linspace(0, np.max(Z), 15)
levels = np.sort(np.concatenate(([a, b], levels)))
axes[0].contour(X, Y, Z, levels=levels, alpha=0.5, colors='grey')
axes[1].contour(X, Y, Z, levels=levels, alpha=0.5, colors='grey')
axes[0].scatter(X_train[:, 0], X_train[:, 1], color='tab:blue', alpha=0.7, label='Samples')

# compute pca directions
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
# plot PCA components
components = pca.components_
mean = pca.mean_
for ii in range(2):
    direction = components[ii]
    t = np.linspace(-8, 2, 100)
    axes[0].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2,
                 label=f'PCA Direction {ii+1}', ls=linestyles[ii])
    axes[1].plot(mean[0] + direction[0] * t, mean[1] + direction[1] * t, color='k', lw=2,
                 label=f'PCA Direction {ii+1}', ls=linestyles[ii])

# plot local PCA data
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)
X_train_pca = pca.inverse_transform(X_train_pca)
axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='tab:blue', alpha=0.5, label='Local PCA Samples')

for ii in range(2):
    axes[ii].set_xlim(lb,ub)
    axes[ii].set_ylim(lb,ub)
    axes[ii].legend(loc='lower right')
    axes[ii].set_xticks([])
    axes[ii].set_yticks([])

axes[0].set_title("Samples")
axes[1].set_title("Local PCA")

plt.tight_layout()
fig.savefig("local_pca_diagram.pdf", format="pdf", bbox_inches="tight")
plt.show()
