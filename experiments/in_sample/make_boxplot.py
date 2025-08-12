import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
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
indir = "./output/"

# load model evaluations
eval_filename = indir + 'diffusion_metrics_local_pca_661.csv'
df_no_pca = pd.read_csv(eval_filename)

# now load the evaluations of QUASR data
baseline_filename = indir + 'baseline_metrics.csv'
df_actuals = pd.read_csv(baseline_filename)
# print(df_actuals.tail(n=5))

""" Clean data """

print("\nSuccess fraction")
print("df_no_pca", df_no_pca['success'].mean())
print("df_actuals", df_actuals['success'].mean())

df_no_pca = df_no_pca.dropna()
df_actuals = df_actuals.dropna()

print("\nNumber of data points")
print("df_no_pca", len(df_no_pca))
print("df_actuals", len(df_actuals))

# compute errror from target
df_no_pca['aspect_percent_error'] = 100 * np.abs(df_no_pca['aspect_ratio'] - df_no_pca['aspect_ratio_condition']) / df_no_pca['aspect_ratio_condition']
df_no_pca['mean_iota_percent_error'] = 100 * np.abs(df_no_pca['mean_iota'] - df_no_pca['mean_iota_condition']) / df_no_pca['mean_iota_condition']
df_no_pca['aspect_error'] = np.abs(df_no_pca['aspect_ratio'] - df_no_pca['aspect_ratio_condition'])
df_no_pca['mean_iota_error'] = np.abs(df_no_pca['mean_iota'] - df_no_pca['mean_iota_condition'])
df_actuals['sqrt_non_qs_error'] = 100 * df_actuals['sqrt_non_qs_error']
df_no_pca['sqrt_non_qs_error']  = 100 * df_no_pca['sqrt_non_qs_error']


""" Box plot """

fig, (ax1,ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3)
width = 0.25

""" Distribution of qs error """

feature = "sqrt_non_qs_error"
left_pos_qs = 1
bplot = ax1.boxplot(df_actuals[feature], whis=(2.5,97.5),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[0], color='black'),
                    widths=width, positions=[left_pos_qs],
                    tick_labels=["All Data"], 
                    label='Actual')
bplot = ax1.boxplot(df_no_pca[feature], whis=(2.5,97.5),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[left_pos_qs+width],
                    tick_labels=[""], 
                    label='DDPM')

xtick_pos = [left_pos_qs]
xtick_label = ["All Data"]
nfp_list = [2,3,4,5,6,7,8]
for ii, nfp_plot in enumerate(nfp_list):    
    # plot mean_iota
    left_pos = left_pos_qs + (ii+1) * width * 3
    label = r'$n_{\text{fp}} = %d$'%nfp_plot
    idx = df_actuals['nfp'] == nfp_plot
    bplot = ax1.boxplot(df_actuals.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[0], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[label])
    idx = df_no_pca['nfp'] == nfp_plot
    bplot = ax1.boxplot(df_no_pca.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos+width],
                        tick_labels=[""])
    xtick_pos.append(left_pos)
    xtick_label.append(label)

ax1.set_ylabel('$J_{QS}$  [%]')
ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax1.legend(loc='lower right', fontsize=10, framealpha=1.0)
xtick = [left_pos_qs+3*width*ii for ii in range(len(nfp_list)+1)]
ax1.set_xticks(xtick_pos,labels=xtick_label, rotation=75)
ax1.set_yscale('log')
ax1.set_yticks([0.1, 1, 10],labels=['0.1', '1', '10'])
ax1.axhline(1.0, color='black', linestyle='--', linewidth=2)
ax1.set_title("Quasisymmetry Error", fontsize=11)


""" plot error in aspect ratio from condition """
feature = 'aspect_percent_error'
left_pos_aspect = 1
bplot = ax2.boxplot(df_no_pca[feature], whis=(2.5,97.5),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[left_pos_aspect],
                    tick_labels=['All Data'], 
                    label='DDPM')


for ii, nfp_plot in enumerate(nfp_list):
    left_pos = left_pos_aspect + (ii+1) * width * 1.5
    idx = df_no_pca['nfp'] == nfp_plot
    bplot = ax2.boxplot(df_no_pca.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[r'$n_{\text{fp}} = %d$'%nfp_plot], 
                        label='DDPM')


ax2.set_ylabel('$J_A$  [%]')
ax2.axhline(5.0, color='black', linestyle='--', linewidth=2)
ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)
# ax2.set_yticks([0, 5, 10, 15, 20], labels=['0', '5', '10', '15', '20'])
ax2.set_ylim(-1,35)
ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(),rotation=75)
ax2.set_title("Error from Aspect Ratio Condition", fontsize=11)




""" plot error in mean iota from condition """
feature = 'mean_iota_percent_error'
left_pos_iota = 1
bplot = ax3.boxplot(df_no_pca[feature], whis=(2.5,97.5),
                    patch_artist=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(facecolor=colors[1], color='black'),
                    widths=width, positions=[left_pos_iota],
                    tick_labels=['All Data'], 
                    label='DDPM')

for ii, nfp_plot in enumerate(nfp_list):    
    # plot mean_iota
    left_pos = left_pos_iota + (ii+1) * width * 1.5
    idx = df_no_pca['nfp'] == nfp_plot
    bplot = ax3.boxplot(df_no_pca.loc[idx, feature], whis=(2.5,97.5),
                        patch_artist=True,
                        medianprops=dict(linewidth=2, color='black'),
                        boxprops=dict(facecolor=colors[1], color='black'),
                        widths=width, positions=[left_pos],
                        tick_labels=[r'$n_{\text{fp}} = %d$'%nfp_plot], 
                        label='DDPM')

ax3.set_xticks(ax3.get_xticks(), ax3.get_xticklabels(),rotation=75)
ax3.set_ylim(-1,35)
ax3.set_ylabel('$J_{\iota}$  [%]')
ax3.axhline(5.0, color='black', linestyle='--', linewidth=2)
ax3.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax3.set_title("Error from Rotational Transform Condition", fontsize=10)

plt.tight_layout()
plt.savefig("./viz/boxplot_in_sample_performance.pdf", bbox_inches='tight', format='pdf')
plt.show()