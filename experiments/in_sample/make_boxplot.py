import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 15})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
outdir = "./viz/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

"""
Make box plots for in-sample performance

Run with:
    python make_boxplot.py
"""

# conditioned on (iota, aspect, nfp, helicity); trained on PCA-50 w/ big model
indir = "./output/"

# load model evaluations
filelist = glob.glob(indir + 'diffusion_metrics_local_pca_661_nfp_*.csv')
filelist.sort()
df_model = pd.DataFrame()
for file in filelist:
    print("Loading", file)
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df['group'] = file.split('_')[-1].split('.')[0]
    df_model = pd.concat([df_model, df], ignore_index=True)

# load the baseline evaluations
filelist = glob.glob(indir + 'baseline_metrics_nfp_*.csv')
filelist.sort()
df_actuals = pd.DataFrame()
for file in filelist:
    print("Loading", file)
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df['group'] = file.split('_')[-1].split('.')[0]
    df_actuals = pd.concat([df_actuals, df], ignore_index=True)

""" Clean data """

print("\nSuccess fraction")
print("df_model", df_model['success'].mean())
print("df_actuals", df_actuals['success'].mean())

df_model = df_model.dropna()
df_actuals = df_actuals.dropna()

print("\nNumber of data points")
print("df_model", len(df_model))
print("df_actuals", len(df_actuals))

# compute errror from target
df_model['aspect_percent_error'] = 100 * np.abs(df_model['aspect_ratio'] - df_model['aspect_ratio_condition']) / df_model['aspect_ratio_condition']
df_model['mean_iota_percent_error'] = 100 * np.abs(df_model['mean_iota'] - df_model['mean_iota_condition']) / df_model['mean_iota_condition']
df_model['aspect_error'] = np.abs(df_model['aspect_ratio'] - df_model['aspect_ratio_condition'])
df_model['mean_iota_error'] = np.abs(df_model['mean_iota'] - df_model['mean_iota_condition'])
df_actuals['sqrt_non_qs_error'] = 100 * df_actuals['sqrt_non_qs_error']
df_model['sqrt_non_qs_error']  = 100 * df_model['sqrt_non_qs_error']


""" Box plot """

fig, (ax1,ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3)
width = 0.25

""" Distribution of qs error """
# sampling group
group_list = ["all",2,3,4,5,6,7,8]

feature = "sqrt_non_qs_error"
left_pos_qs = 1

# storage
xtick_pos = []
xtick_label = []

for ii, group in enumerate(group_list):
    # select the subset of data
    idx = (df_actuals['group'] == str(group))

    if group == "all":
        tlabel = 'All Data'
    else:
        tlabel = r'$n_{\text{fp}} = %s$'%group

    # legend labels
    if ii == 0:
        model_label = "DDPM"
        actual_label = "QUASR"
    else:
        model_label = None
        actual_label = None

    left_pos = left_pos_qs + (ii) * width * 3

    bplot = ax1.boxplot(df_actuals.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[0], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[tlabel],
                        label = [actual_label])
    
    idx = (df_model['group'] == str(group))
    bplot = ax1.boxplot(df_model.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos+width],
                        tick_labels=[""],
                        label = [model_label])
    xtick_pos.append(left_pos)
    xtick_label.append(tlabel)

ax1.set_ylabel('$J_{QS}$  [%]')
ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax1.legend(loc='lower right', fontsize=12, framealpha=1.0)
# xtick = [left_pos_qs+3*width*ii for ii in range(len(nfp_list)+1)]
ax1.set_xticks(xtick_pos,labels=xtick_label, rotation=75)
ax1.set_yscale('log')
ax1.set_yticks([0.1, 1, 10],labels=['0.1', '1', '10'])
ax1.axhline(1.0, color='black', linestyle='--', linewidth=2)
# ax1.set_title("Quasisymmetry Error", fontsize=12)


""" plot error in aspect ratio from condition """

feature = 'aspect_percent_error'
left_pos_aspect = 1

for ii, group in enumerate(group_list):
    idx = df_model['group'] == str(group)

    if group == "all":
        label = "All Data"
    else:
        label = r'$n_{\text{fp}} = %s$'%group

    left_pos = left_pos_aspect + (ii+1) * width * 1.5
    bplot = ax2.boxplot(df_model.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[label], 
                        label='DDPM')


ax2.set_ylabel('$c_A$  [%]')
ax2.axhline(5.0, color='black', linestyle='--', linewidth=2)
ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)
# ax2.set_yticks([0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'])
ax2.set_ylim(-1,35)
ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(),rotation=75)
# ax2.set_title("Error from Aspect Ratio Condition", fontsize=12)




""" plot error in mean iota from condition """
feature = 'mean_iota_percent_error'

left_pos_aspect = 1

for ii, group in enumerate(group_list):
    idx = df_model['group'] == str(group)

    if group == "all":
        label = "All Data"
    else:
        label = r'$n_{\text{fp}} = %s$'%group

    left_pos = left_pos_aspect + (ii+1) * width * 1.5
    bplot = ax3.boxplot(df_model.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[label], 
                        label='DDPM')

ax3.set_xticks(ax3.get_xticks(), ax3.get_xticklabels(),rotation=75)
ax3.set_ylim(-1,35)
ax3.set_ylabel('$c_{\iota}$  [%]')
ax3.axhline(5.0, color='black', linestyle='--', linewidth=2)
ax3.grid(color='lightgray', linestyle='--', linewidth=0.5)
# ax3.set_title("Error from Rotational Transform Condition", fontsize=12)

plt.tight_layout()
plt.savefig("./viz/boxplot_in_sample_performance.pdf", bbox_inches='tight', format='pdf')
plt.show()