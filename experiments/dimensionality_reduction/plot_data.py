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
plt.rcParams.update({'font.size': 11})
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
df_plt = df_plt.loc[:, ['n_pca', 'qs_error', 'iota', 'aspect_ratio']]


""" Plot the relative error in statistics """


# take means
df_mean = df_plt.groupby('n_pca').mean().reset_index()
df_mean = df_mean.rename(columns={'n_pca': 'n_pca', 
                                   'qs_error': 'mean_qs_error', 
                                   'iota': 'mean_iota', 
                                   'aspect_ratio': 'mean_aspect_ratio'})
# compute variances
df_std = df_plt.groupby('n_pca').std().reset_index()
df_std = df_std.rename(columns={'n_pca': 'n_pca', 
                                   'qs_error': 'std_qs_error', 
                                   'iota': 'std_iota', 
                                   'aspect_ratio': 'std_aspect_ratio'})
# concatenate means and stds
df_mean = pd.merge(df_mean, df_std, on='n_pca', how='outer')

# compute relative error
cols = ['mean_qs_error', 'mean_iota', 'mean_aspect_ratio']
df_target = df_mean.loc[df_mean.n_pca == 661].squeeze() # convert to series
new_cols = ['mean_qs_error_rel_error', 'mean_iota_rel_error', 'mean_aspect_ratio_rel_error']
df_mean.loc[:, new_cols] = 100 * np.abs(((df_mean[cols] - df_target[cols]) / df_target[cols]).values)

cols = ['std_qs_error', 'std_iota', 'std_aspect_ratio']
df_target = df_mean.loc[df_mean.n_pca == 661].squeeze() # convert to series
new_cols = ['std_qs_error_rel_error', 'std_iota_rel_error', 'std_aspect_ratio_rel_error']
df_mean.loc[:, new_cols] = 100 * np.abs(((df_mean[cols] - df_target[cols]) / df_target[cols]).values)


# plot means
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(df_mean.n_pca, df_mean.mean_qs_error_rel_error, label='QS Error', lw=2, ls='dashdot', marker='o', color=colors[0])
ax[0].plot(df_mean.n_pca, df_mean.mean_iota_rel_error, label='Rotational Transform', lw=2, ls='dotted', marker='s', color=colors[1])
ax[0].plot(df_mean.n_pca, df_mean.mean_aspect_ratio_rel_error, label='Aspect Ratio', lw=2, ls='--', marker='*', color=colors[2])
ax[0].axvline(50, color='k', ls='--', lw=2)
ax[0].set_xlabel('Number of PCA components')
ax[0].set_ylabel('Relative Error [%]')
ax[0].set_xscale('log')
ax[0].legend(loc='upper right', fontsize=9)
# ax[0].set_title('Relative Error in Mean Values')
ax[0].grid(True, zorder=0, color='lightgray')


# # plot standard deviations
# ax[1].plot(df_mean.n_pca, df_mean.std_qs_error_rel_error, label='QS Error', lw=2, marker='o')
# ax[1].plot(df_mean.n_pca, df_mean.std_iota_rel_error, label='Rotational Transform', lw=2, ls='-', marker='s')
# ax[1].plot(df_mean.n_pca, df_mean.std_aspect_ratio_rel_error, label='Aspect Ratio', lw=2, ls='-', marker='*')
# ax[1].axvline(50, color='k', ls='--', lw=2)
# ax[1].set_xlabel('Number of PCA components')
# ax[1].set_ylabel('Relative Error [%]')
# ax[1].set_xscale('log')
# ax[1].legend(loc='upper right', fontsize=10)
# ax[1].set_title('Relative Error in Standard Deviations')
# ax[1].grid(True, zorder=0, color='lightgray')

# kde plot
sizes = [661, 50]
# cmaps = ['Blues', 'Reds']
cmaps = ['Greys', 'Blues']
colors = ['lightgray', 'steelblue']
linestyles = ['solid', 'solid']
alphas = [1.0, 0.8]
for ii, n_pca in enumerate(sizes):
    idx= (df_plt.n_pca == n_pca)
    X = df_plt.loc[idx, ['iota', 'qs_error']].values
    X[:,1] *= 100 # scale QS error to percentage
    kde = gaussian_kde(X.T)
    x_min, x_max = 1.2*X[:, 0].min(), X[:, 0].max()
    y_min, y_max = 0.0, X[:, 1].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100), indexing='ij')
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = kde(positions).reshape(x_grid.shape)

    # ax[-1].contour(x_grid, y_grid, density, cmap=cmaps[ii], alpha=1.0, linewidths=2, linestyles=linestyles[ii])
    ax[-1].contour(x_grid, y_grid, density, colors=colors[ii], alpha=alphas[ii], linewidths=3, linestyles=linestyles[ii])


ax[-1].set_ylabel('QS Error [%]')
ax[-1].set_xlabel('Rotational Transform')
ax[-1].set_ylim(0, 20.0)
ax[-1].set_xlim(y_min, 2.5)
ax[-1].set_yticks([0.0, 5, 10, 15, 20])
ax[-1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax[-1].grid(zorder=0, color='lightgray')
# legend
from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor=plt.get_cmap(cmaps[0])(0.4), label='$n_{pca}=%d$'%sizes[0]),
#                    Patch(facecolor=plt.get_cmap(cmaps[1])(0.4), label='$n_{pca}=%d$'%sizes[1])]
legend_elements = [Patch(facecolor=colors[0], label='$n_{pca}=%d$'%sizes[0]),
                   Patch(facecolor=colors[1], label='$n_{pca}=%d$'%sizes[1])]
ax[-1].legend(handles=legend_elements, loc='upper left', fontsize=9)


plt.tight_layout()
plt.savefig("./dimensionality_reduction_plot.pdf", format="pdf")
plt.show()
