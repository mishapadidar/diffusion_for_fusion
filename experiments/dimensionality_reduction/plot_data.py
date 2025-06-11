import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
# import seaborn as sns
from scipy.stats import gaussian_kde



""" 
Plot the loss of information caused by dimensionality reduction using PCA.
"""

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['salmon', 'goldenrod', 'mediumseagreen']



filelist = glob("./output/evaluations_pca_*.pkl")

df_plt = pd.read_pickle(filelist[0])

for ii, ff in enumerate(filelist[1:]):
    df = pd.read_pickle(ff)
    
    # concatenate the dataframes
    df_plt = pd.concat([df_plt, df], ignore_index=True)

# drop data
df_plt = df_plt.loc[:, ['n_pca', 'qs_error', 'aspect_ratio', 'ID']]

# convert qs_error to percentage
df_plt['qs_error'] = df_plt['qs_error'] * 100  # convert to percentage
# convert qs_error to percentage
df_plt['log_qs_error'] = np.log10(df_plt['qs_error'])

# drop n_pca < 5
df_plt = df_plt.loc[df_plt.n_pca >= 6].reset_index(drop=True)

""" Plot the relative error in statistics """

# make a columns with the actual values
df_actual = df_plt.loc[df_plt.n_pca == 661]
df_plt = df_plt.merge(df_actual, on='ID', suffixes=('', '_actual'), how='left')

# compute relative error
cols = ['log_qs_error', 'aspect_ratio']
mins = df_plt[[f'{col}_actual' for col in cols]].values
df_plt.loc[:, ['log_qs_error_rel_error', 'aspect_ratio_rel_error']] = 100 * np.abs((df_plt[cols] - mins) / mins).values


fig, ax = plt.subplots(1, 3, figsize=(12, 4))


# plot the PCA variance explained
X = np.load('../../data/dofs.npy') # x-values
C = np.cov(X.T)
eig = np.sort(np.linalg.eigvals(C))[::-1]
eig = eig / np.sum(eig) * 100 
var = np.cumsum(eig) 
ax[0].plot(1+np.arange(len(eig)), var, lw=2, color=colors[0], ls='-.')
# plot just a handful of markers
idx = np.logspace(0, np.log10(len(eig)), num=20, dtype=int) - 1  # log spaced indices
ax[0].scatter((1+np.arange(len(eig)))[idx], var[idx], s=40, color=colors[0], marker='*', ls='-.')
ax[0].axvline(50, color='k', ls='--', lw=2)
ax[0].set_xlabel('Number of Principal Components')
ax[0].set_ylabel("% of Variance Explained")
ax[0].set_xscale('log')
ax[0].grid(zorder=0, color='lightgray')
ax[0].set_xticks([1, 10, 100, 1000])
ax[0].set_yticks([60, 70, 80, 90, 100])

# plot relative error
df_stats = df_plt.groupby('n_pca')['aspect_ratio_rel_error'].agg(['median', 'max']).reset_index()
df_stats.rename(columns={'median': 'median_aspect_ratio_rel_error', 'max': 'max_aspect_ratio_rel_error' }, inplace=True)
ax[1].plot(df_stats.n_pca, df_stats.median_aspect_ratio_rel_error, label='Aspect Ratio', lw=2, ls='--', marker='*', color=colors[2])

# border where max(aspect ratio relative error) > 20 %
x = np.concatenate(([0], df_stats.n_pca.values))
y = np.concatenate(([np.inf], df_stats.max_aspect_ratio_rel_error.values, ))
ax[1].fill_between(x, 0, 1, where=y > 20,
                color='lightgrey', alpha=0.5, transform=ax[1].get_xaxis_transform())


df_stats = df_plt.groupby('n_pca')['log_qs_error_rel_error'].agg(['median', 'max']).reset_index()
df_stats.rename(columns={'median': 'median_log_qs_error_rel_error'}, inplace=True)
ax[1].plot(df_stats.n_pca, df_stats.median_log_qs_error_rel_error, label='$\log$(QS-Error)', lw=2, ls='--', marker='*', color=colors[1])

ax[1].axvline(50, color='k', ls='--', lw=2)
ax[1].set_xlabel('Number of Principal Components')
ax[1].set_ylabel('Median Percentage Error')
ax[1].set_xscale('log')
ax[1].legend(loc='upper right', fontsize=9)
ax[1].grid(True, zorder=0, color='lightgray')


# kde plot
sizes = [661, 50]
# cmaps = ['Blues', 'Reds']
cmaps = ['Greys', 'Blues']
colors = ['lightgray', 'steelblue']
linestyles = ['solid', 'solid']
alphas = [1.0, 0.8]
for ii, n_pca in enumerate(sizes):
    idx= (df_plt.n_pca == n_pca)
    X = df_plt.loc[idx, ['aspect_ratio', 'qs_error']].values
    kde = gaussian_kde(X.T)
    x_min, x_max = 1.0, X[:, 0].max()
    y_min, y_max = 0.0, X[:, 1].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100), indexing='ij')
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = kde(positions).reshape(x_grid.shape)

    if ii == 0:
        levels = np.logspace(-6, np.log10(np.max(density)/20), 5, endpoint=True)
        levels = np.sort(np.concatenate((np.linspace(np.min(density), np.max(density),20, endpoint=True), levels)))
    else:
        levels = np.logspace(-6, np.log10(np.max(density)/20), 5, endpoint=True)
        levels = np.sort(np.concatenate((np.linspace(np.min(density), np.max(density),10, endpoint=True), levels)))
    ax[-1].contour(x_grid, y_grid, density, levels=levels, colors=colors[ii], alpha=alphas[ii], linewidths=2, linestyles=linestyles[ii])

ax[-1].set_ylabel('QS Error [%]')
ax[-1].set_xlabel('Aspect Ratio')
ax[-1].set_ylim(0, 20.0)
# ax[-1].set_xlim(y_min, 2.5)
ax[-1].set_yticks([0.0, 5, 10, 15, 20])
ax[-1].set_xticks([1, 5.0, 10, 15, 20, 25])
ax[-1].grid(zorder=0, color='lightgray')
# legend
from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor=plt.get_cmap(cmaps[0])(0.4), label='$n_{pca}=%d$'%sizes[0]),
#                    Patch(facecolor=plt.get_cmap(cmaps[1])(0.4), label='$n_{pca}=%d$'%sizes[1])]
legend_elements = [Patch(facecolor=colors[0], label='$n_r=n_{\mathbf{x}}$'),
                   Patch(facecolor=colors[1], label='$n_r=%d$'%sizes[1])]
ax[-1].legend(handles=legend_elements, loc='upper left', fontsize=9)


plt.tight_layout()
plt.savefig("./dimensionality_reduction_plot.pdf", format="pdf")
plt.show()
