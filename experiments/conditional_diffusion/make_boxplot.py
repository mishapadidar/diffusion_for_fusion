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
print(df_model.tail(n=5))

# load model evaluations
eval_filename = indir + "evaluations/" + f'diffusion_metrics_local_pca_5.csv'
df_model2 = pd.read_csv(eval_filename)

# now load the evaluations of QUASR data
baseline_filename = indir + "evaluations/" + f'baseline_metrics.csv'
df_actuals = pd.read_csv(baseline_filename)
print(df_actuals.tail(n=5))


# """ Clean data """

# # # # drop outliers that could be easily discarded
# df_model = df_model[df_model['sqrt_qs_error'] < 0.5]
# df_model2 = df_model2[df_model2['sqrt_qs_error'] < 0.5]
# df_actuals = df_actuals[df_actuals['sqrt_qs_error'] < 0.5]

# df_model = df_model[(df_model['nfp_condition'] ==4) ]
# df_model2 = df_model2[(df_model2['nfp_condition'] ==4)]
# df_actuals = df_actuals[(df_actuals['nfp'] == 4)]

# compute errror from target
df_model['aspect_error'] = np.abs(df_model['aspect_ratio'] - df_model['aspect_ratio_condition']) / df_model['aspect_ratio_condition']
df_model2['aspect_error'] = np.abs(df_model2['aspect_ratio'] - df_model2['aspect_ratio_condition']) / df_model2['aspect_ratio_condition']

""" box plot """

fig, (ax1,ax2) = plt.subplots(figsize=(12, 6), ncols=2)

# plot distribution of qs error
ax1.boxplot(df_actuals['sqrt_qs_error'], whis=(5,95), positions=[1], tick_labels=['QUASR'])
ax1.boxplot(df_model['sqrt_qs_error'], whis=(5,95), positions=[2], tick_labels=['$n_{local}$ = %d'% n_local_pca])
ax1.boxplot(df_model2['sqrt_qs_error'], whis=(5,95), positions=[3], tick_labels=['$n_{local}$ = 2'])
ax1.set_ylabel('$J_{QS}$ [%]')

# plot error in aspect ratio from condition
ax2.boxplot(df_model['aspect_error'], whis=(5,95), positions=[2], tick_labels=['$n_{local}$ = %d'% n_local_pca])
ax2.boxplot(df_model2['aspect_error'], whis=(5,95), positions=[3], tick_labels=['$n_{local}$ = 2'])
ax2.set_ylabel('Percent Error in Aspect Ratio')
ax2.axhline(0.0, color='black', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()

