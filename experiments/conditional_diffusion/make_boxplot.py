import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['salmon', 'goldenrod', 'mediumseagreen']
outdir = "./viz/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

"""
Make box plots for in-sample performance

Two goals of the box plot:
- show that diffusion model faithfully reproduces condition values (aspect ratio, iota)
- show that local PCA helps reduce error in the model (qs error, aspect ratio error, iota error)

We can analyze performance on entire dataset, and using subsets

"""

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "output/mean_iota_aspect_ratio_nfp_helicity/run_uuid_0278f98c-aaff-40ce-a7cd-b21a6fac5522/"
n_local_pca = 661

# load model evaluations
eval_filename = indir + "evaluations/" + f'diffusion_metrics_local_pca_{n_local_pca}.csv'
df_model = pd.read_csv(eval_filename)
# print(df_model.tail(n=5))

# load model evaluations
eval_filename = indir + "evaluations/" + f'diffusion_metrics_local_pca_5.csv'
df_model2 = pd.read_csv(eval_filename)

# now load the evaluations of QUASR data
baseline_filename = indir + "evaluations/" + f'baseline_metrics.csv'
df_actuals = pd.read_csv(baseline_filename)
# print(df_actuals.tail(n=5))

# """ Clean data """

print("\nSuccess fraction")
print("df_model", df_model['success'].mean())
print("df_model2", df_model2['success'].mean())
print("df_actuals", df_actuals['success'].mean())

df_model = df_model.dropna()
df_model2 = df_model2.dropna()
df_actuals = df_actuals.dropna()

print("\nNumber of data points")
print("df_model", df_model.shape[0])
print("df_model2", df_model2.shape[0])
print("df_actuals", df_actuals.shape[0])

# # # # drop outliers that could be easily discarded
# df_model = df_model[df_model['sqrt_qs_error'] < 0.4]
# df_model2 = df_model2[df_model2['sqrt_qs_error'] < 0.4]
# df_actuals = df_actuals[df_actuals['sqrt_qs_error'] < 0.4]

# df_model = df_model[(df_model['nfp_condition'] ==4) & (df_model['helicity_condition'] == 1)]
# df_model2 = df_model2[(df_model2['nfp_condition'] ==4) & (df_model2['helicity_condition'] == 1)]
# df_actuals = df_actuals[(df_actuals['nfp'] == 4) & (df_actuals['helicity'] == 1)]

# compute errror from target
df_model['aspect_error'] = 100 * np.abs(df_model['aspect_ratio'] - df_model['aspect_ratio_condition']) / df_model['aspect_ratio_condition']
df_model['iota_error'] = 100 * np.abs(df_model['iota'] - df_model['iota_condition']) / df_model['iota_condition']
df_model2['aspect_error'] = 100 * np.abs(df_model2['aspect_ratio'] - df_model2['aspect_ratio_condition']) / df_model2['aspect_ratio_condition']
df_model2['iota_error'] = 100 * np.abs(df_model2['iota'] - df_model2['iota_condition']) / df_model2['iota_condition']

# log(qs error)
df_actuals['log_sqrt_qs_error'] = np.log(df_actuals['sqrt_qs_error'])
df_model['log_sqrt_qs_error']  =   np.log(df_model['sqrt_qs_error'])
df_model2['log_sqrt_qs_error'] =  np.log(df_model2['sqrt_qs_error'])

# convert to percent
df_actuals['sqrt_qs_error'] = 100 * df_actuals['sqrt_qs_error']
df_model['sqrt_qs_error'] = 100 * df_model['sqrt_qs_error']
df_model2['sqrt_qs_error'] = 100 * df_model2['sqrt_qs_error']

# compute sqrt(qs_error_2term)
df_actuals['sqrt_qs_error_2term'] = np.sqrt(df_actuals['qs_error_2term'])
df_model['sqrt_qs_error_2term']   = np.sqrt(df_model['qs_error_2term'])
df_model2['sqrt_qs_error_2term']  = np.sqrt(df_model2['qs_error_2term'])

print("\nMean and std of qs error")
print("df_model", df_model['sqrt_qs_error'].mean(), df_model['sqrt_qs_error'].std())
print("df_model2", df_model2['sqrt_qs_error'].mean(), df_model2['sqrt_qs_error'].std())
print("df_actuals", df_actuals['sqrt_qs_error'].mean(), df_actuals['sqrt_qs_error'].std())


from scipy.stats import wasserstein_distance
print("\nWasserstein distance of qs error")
print("df_model", wasserstein_distance(df_model['sqrt_qs_error'], df_actuals['sqrt_qs_error']))
print("df_model2", wasserstein_distance(df_model2['sqrt_qs_error'], df_actuals['sqrt_qs_error']))

print("\nWasserstein distance of iota")
print("df_model", wasserstein_distance(df_model['iota'], df_actuals['iota']))
print("df_model2", wasserstein_distance(df_model2['iota'], df_actuals['iota']))

print("\nWasserstein distance of aspect ratio")
print("df_model", wasserstein_distance(df_model['aspect_ratio'], df_actuals['aspect_ratio']))
print("df_model2", wasserstein_distance(df_model2['aspect_ratio'], df_actuals['aspect_ratio']))

print("\nNumber of field periods")
print(df_model.nfp_condition.value_counts())

""" box plot """

fig, (ax1,ax2) = plt.subplots(figsize=(12, 6), ncols=2)
width = 0.25

# plot distribution of qs error

# bplot = ax1.boxplot(df_actuals['sqrt_qs_error'], whis=(5,95),
bplot = ax1.boxplot(df_actuals['log_sqrt_qs_error'], whis=(5,95),
# bplot = ax1.boxplot(df_actuals['sqrt_qs_error_2term'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[0], color='black'),
                    widths=width, positions=[1],
                    tick_labels=['QUASR'])
bplot = ax1.boxplot(df_model['log_sqrt_qs_error'], whis=(5,95),
# bplot = ax1.boxplot(df_model['sqrt_qs_error'], whis=(5,95),
# bplot = ax1.boxplot(df_model['sqrt_qs_error_2term'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[2],
                    tick_labels=['$n_{local}$ = %d'% n_local_pca])
bplot = ax1.boxplot(df_model2['log_sqrt_qs_error'], whis=(5,95),
# bplot = ax1.boxplot(df_model2['sqrt_qs_error'], whis=(5,95),
# bplot = ax1.boxplot(df_model2['sqrt_qs_error_2term'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[2], color='black'),
                    widths=width, positions=[3],
                    tick_labels=['$n_{local}$ = 5'])
ax1.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax1.set_ylabel('$J_{QS}$ [%]')

# plot error in aspect ratio from condition
bplot = ax2.boxplot(df_model['aspect_error'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[2],
                    tick_labels=['Aspect Ratio'], 
                    label='$n_{local}$ = %d'% n_local_pca)
bplot = ax2.boxplot(df_model2['aspect_error'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[2], color='black'),
                    widths=width, positions=[2+width],
                    tick_labels=[""], 
                    label='$n_{local}$ = 5')
bplot = ax2.boxplot(df_model['iota_error'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[3],
                    tick_labels=['Rotational Transform'])
bplot = ax2.boxplot(df_model2['iota_error'], whis=(5,95),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[2], color='black'),
                    widths=width, positions=[3+width],
                    tick_labels=[""])
ax2.set_ylabel('Deviation from condition [%]')
ax2.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax2.legend(loc='upper right')

# # fill with colors
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

plt.tight_layout()
plt.show()

