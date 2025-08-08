import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
colors = ['goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]

outdir = "./viz/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

"""
Make box plots for out-of-sample performance

Show that diffusion model faithfully reproduces condition values (aspect ratio, iota)

"""


# load model evaluations
filelist = ["./output/diffusion_metrics_iota_0.36_nfp_2_helicity_1_aspect_ratio_4.5.csv",
            "./output/diffusion_metrics_iota_0.5_nfp_3_helicity_1_aspect_ratio_9.0.csv",
            "./output/diffusion_metrics_iota_1.4_nfp_4_helicity_1_aspect_ratio_11.0.csv"]

df_list = [pd.read_csv(ff) for ff in filelist]


""" Clean data """
labels = []

n_dfs = len(df_list)
for ii in range(n_dfs):
    print(f"\nDataframe {ii+1}/{n_dfs}")
    print("Success fraction", df_list[ii]['success'].mean())

    df_list[ii] = df_list[ii].dropna()

    print("Number of data points", len(df_list[ii]))

    # compute errror from target
    df_list[ii]['aspect_percent_error'] = 100 * np.abs(df_list[ii]['aspect_ratio'] - df_list[ii]['aspect_ratio_condition']) / df_list[ii]['aspect_ratio_condition']
    df_list[ii]['mean_iota_percent_error'] = 100 * np.abs(df_list[ii]['mean_iota'] - df_list[ii]['mean_iota_condition']) / df_list[ii]['mean_iota_condition']
    df_list[ii]['aspect_error'] = np.abs(df_list[ii]['aspect_ratio'] - df_list[ii]['aspect_ratio_condition'])
    df_list[ii]['mean_iota_error'] = np.abs(df_list[ii]['mean_iota'] - df_list[ii]['mean_iota_condition'])

    # log(qs error)
    df_list[ii]['log_sqrt_qs_error_boozer'] =  np.log10(df_list[ii]['sqrt_qs_error_boozer'])
    df_list[ii]['log_sqrt_qs_error_2term'] =  np.log10(df_list[ii]['sqrt_qs_error_2term'])
    df_list[ii]['log_sqrt_non_qs_error'] =  np.log10(df_list[ii]['sqrt_non_qs_error'])

    # # convert to percent
    df_list[ii]['sqrt_qs_error_boozer']  = 100 * df_list[ii]['sqrt_qs_error_boozer']
    df_list[ii]['sqrt_qs_error_2term']  = 100 * df_list[ii]['sqrt_qs_error_2term']
    df_list[ii]['sqrt_non_qs_error']  = 100 * df_list[ii]['sqrt_non_qs_error']

    # label = r"$n_{\text{fp}} =%d$"%(df_list[ii]['nfp'].iloc[0])
    label = "config %d"%(ii+1)
    labels.append(label)
""" Box plot """

fig, (ax1,ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3)
width = 0.25

""" Distribution of qs error """

feature = "sqrt_non_qs_error"
left_pos_qs = 1

for ii, df in enumerate(df_list):

    bplot = ax1.boxplot(df[feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[ii], color='black'),
                        widths=width, positions=[left_pos_qs+width*ii],
                        tick_labels=[""], 
                        label=labels[ii])

ax1.set_ylabel('$J_{QS}$  [%]')
ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax1.legend(loc='lower right', fontsize=10, framealpha=1.0)
# xtick = [left_pos_qs+3*width*ii for ii in range(len(nfp_list)+1)]
# ax1.set_xticks(xtick_pos,labels=xtick_label, rotation=60)
ax1.set_yscale('log')
ax1.set_yticks([0.1, 1, 10],labels=['0.1', '1', '10'])
ax1.set_title("Quasisymmetry Error", fontsize=11)


# """ plot error in aspect ratio from condition """
feature = 'aspect_percent_error'
left_pos_qs = 1

for ii, df in enumerate(df_list):

    bplot = ax2.boxplot(df[feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[ii], color='black'),
                        widths=width, positions=[left_pos_qs+width*ii],
                        tick_labels=[""], 
                        label=labels[ii])

ax2.set_ylabel('Error [%]')
ax2.axhline(0.0, color='black', linestyle='--', linewidth=2)
# ax2.legend(loc='upper right')
ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)
# print(ax2.get_xticks())
# print(ax2.get_xticklabels())
# ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(),rotation=60)
ax2.set_title("Error from Aspect Ratio Condition", fontsize=11)



""" plot error in mean iota from condition """
feature = 'mean_iota_error'
left_pos_qs = 1

for ii, df in enumerate(df_list):

    bplot = ax3.boxplot(df[feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[ii], color='black'),
                        widths=width, positions=[left_pos_qs+width*ii],
                        tick_labels=[""], 
                        label=labels[ii])

# ax3.set_xticks(ax3.get_xticks(), ax3.get_xticklabels(),rotation=60)

ax3.set_ylim(-0.03, 0.25)
ax3.set_ylabel('Error')
ax3.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax3.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax3.set_title("Error from Rotational Transform Condition", fontsize=10)

plt.tight_layout()
plt.savefig("./viz/boxplot_out_of_sample_performance.pdf", bbox_inches='tight', format='pdf')
plt.show()