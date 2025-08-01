import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange']
markers = ['o', 's', 'x', '^', '*', 'p', 'D', 'v', '>', '<',  'h']


"""
Plot the distribution of data in QUASR, and show where we will perform out of sample evaluation.
"""


# load quasr data
infile = "../../data/QUASR.pkl"
df = pd.read_pickle(infile)
print(df.columns)

# round iota and aspect to pretty up the plot
df['mean_iota'] = np.round(df['mean_iota'], 1)
df['aspect_ratio'] = np.round(df['aspect_ratio'], 0)

# give each nfp helicity pair an ID
df['nfp_helicity'] = "(" + df['nfp'].astype(str) + ',' + df['helicity'].astype(str) + ")"

# # drop 'nfp_helicity' if less than 10000 samples
# df_counts = df.groupby('nfp_helicity').size().reset_index(name='counts')
# df_counts = df_counts[df_counts['counts'] >= 10000]
# df = df[df['nfp_helicity'].isin(df_counts['nfp_helicity'])]

# get unique (iota, aspect, nfp, helicity) combinations
df_unique = df[['mean_iota', 'aspect_ratio', 'nfp', 'helicity']].drop_duplicates()

markersize=25
alpha = 0.7

fig, ax = plt.subplots(figsize=(8, 6))
for helicity in df_unique['helicity'].unique():
    idx = (df_unique.helicity == helicity)
    mean_iota = df_unique['mean_iota'][idx]
    aspect_ratio = df_unique['aspect_ratio'][idx]
    if helicity == 1:
        qs_label = 'QH'
    else:
        qs_label = 'QA'
    plt.scatter(mean_iota, aspect_ratio, c=colors[helicity], s=markersize, marker=markers[helicity], alpha=alpha, label='%s'%(qs_label))

# scatter some misc data points
plt.scatter([0.36], [4.5], color=colors[2], s=int(2*markersize), marker=markers[4], alpha=1.0, label=r'$n_{\text{fp}}=2$ QA')
plt.scatter([0.5], [9.0], color=colors[2], s=int(2*markersize), marker=markers[2], alpha=1.0, label=r'$n_{\text{fp}}=3$ QA')
plt.scatter([1.4], [11.0], color=colors[2], s=int(2*markersize), marker=markers[5], alpha=1.0, label=r'$n_{\text{fp}}=4$ QH')
plt.scatter([2.5], [17.0], color=colors[2], s=int(2*markersize), marker=markers[3], alpha=1.0, label=r'$n_{\text{fp}}=5$ QH')

ax.set_xlabel('Mean Rotational Transform')
ax.set_ylabel('Aspect Ratio')
plt.legend(loc='upper right', fontsize=10, framealpha=1.0)
plt.show()