import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 14})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

sns.set_theme(style="ticks", font='serif')
Y = pd.read_pickle('../../data/QUASR.pkl') # y-values
df = Y[['mean_iota', 'aspect_ratio', 'nfp', 'helicity']]


df.rename(columns={'mean_iota': 'Mean $\iota$', 'aspect_ratio': 'Aspect Ratio', 'nfp': r'$n_{fp}$', 'helicity': 'Helicity'}, inplace=True)
# df = df.sample(n=10000, axis=0)

plot = sns.pairplot(df, hue="Helicity")
fig = plot.figure
fig.suptitle('QUASR Data Distribution')
sns.move_legend(plot, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()