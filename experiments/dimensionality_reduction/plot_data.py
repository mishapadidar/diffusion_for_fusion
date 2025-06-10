import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

""" 
Plot the loss of information caused by dimensionality reduction using PCA.
"""

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 11})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ["tab:blue", 'darkorange', 'forestgreen', 'firebrick']



filelist = glob("./output/evaluations_pca_*.pkl")

df_plt = pd.read_pickle(filelist[0])

for ii, ff in enumerate(filelist[1:]):
    df = pd.read_pickle(ff)
    
    # concatenate the dataframes
    df_plt = pd.concat([df_plt, df], ignore_index=True)

# drop data
df_plt = df_plt.loc[:, ['n_pca', 'qs_error', 'iota', 'aspect_ratio']]

# """ Make a Q-Q plot """
# print(df_plt)

# q_list = np.linspace(0, 1, 30)
# df_qq = df_plt.groupby('n_pca').quantile(q_list).reset_index()
# df_qq = df_qq.rename(columns={'level_1': 'quantile',})
# df_qq.sort_values(by=['n_pca', 'quantile'], inplace=True)

# x0 = df_qq.loc[df_qq.n_pca == 661, 'qs_error'].values
# for n_pca in df_qq.n_pca.unique():
#     x = df_qq.loc[df_qq.n_pca == n_pca, 'qs_error'].values
#     plt.plot(x0, x)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()
# quit()

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
print(df_mean)

# compute relative error
cols = ['mean_qs_error', 'mean_iota', 'mean_aspect_ratio']
df_target = df_mean.loc[df_mean.n_pca == 661].squeeze() # convert to series
new_cols = ['mean_qs_error_rel_error', 'mean_iota_rel_error', 'mean_aspect_ratio_rel_error']
df_mean.loc[:, new_cols] = 100 * np.abs(((df_mean[cols] - df_target[cols]) / df_target[cols]).values)

cols = ['std_qs_error', 'std_iota', 'std_aspect_ratio']
df_target = df_mean.loc[df_mean.n_pca == 661].squeeze() # convert to series
new_cols = ['std_qs_error_rel_error', 'std_iota_rel_error', 'std_aspect_ratio_rel_error']
df_mean.loc[:, new_cols] = 100 * np.abs(((df_mean[cols] - df_target[cols]) / df_target[cols]).values)
print(df_mean)

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(df_mean.n_pca, df_mean.mean_qs_error_rel_error, label='QS Error', lw=2, marker='o')
ax[0].plot(df_mean.n_pca, df_mean.mean_iota_rel_error, label='Rotational Transform', lw=2, ls='--', marker='s')
ax[0].plot(df_mean.n_pca, df_mean.mean_aspect_ratio_rel_error, label='Aspect Ratio', lw=2, ls='-.', marker='*')
ax[0].set_xlabel('Number of PCA components')
ax[0].set_ylabel('Relative Error [%]')
ax[0].set_xscale('log')
# ax[0].legend(loc='upper right')
ax[0].set_title('Relative Error in Mean Values')
ax[0].grid(True, zorder=0, color='lightgray')

ax[1].plot(df_mean.n_pca, df_mean.std_qs_error_rel_error, label='QS Error', lw=2, marker='o')
ax[1].plot(df_mean.n_pca, df_mean.std_iota_rel_error, label='Rotational Transform', lw=2, ls='--', marker='s')
ax[1].plot(df_mean.n_pca, df_mean.std_aspect_ratio_rel_error, label='Aspect Ratio', lw=2, ls='-.', marker='*')
ax[1].set_xlabel('Number of PCA components')
ax[1].set_ylabel('Relative Error [%]')
ax[1].set_xscale('log')
ax[1].legend(loc='upper right')
ax[1].set_title('Relative Error in Standard Deviations')
ax[1].grid(True, zorder=0, color='lightgray')

plt.tight_layout()
plt.show()