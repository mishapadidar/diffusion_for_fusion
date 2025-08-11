import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 13})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['lightcoral', 'goldenrod', "cornflowerblue", "mediumorchid", 'mediumseagreen','orange']
markers = ['s','o',  'x', '^', '*', 'p', 'D', '+', 'v', '1', '>', '<',  'h']
outdir = "./viz/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

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

# get unique (iota, aspect, nfp, helicity) combinations
idx = df[['mean_iota', 'aspect_ratio']].duplicated()
df_unique = df.loc[~idx, ['mean_iota', 'aspect_ratio', 'nfp', 'helicity']]#.drop_duplicates()


markersize=30
alpha = 1.0

fig, (ax, ax2) = plt.subplots(figsize=(12, 6), ncols=2, gridspec_kw={'width_ratios': [3, 1]})
ax.grid(color='lightgray', linestyle='--', linewidth=0.5, zorder=0)

""" Plot the dataset """

for helicity in df_unique['helicity'].unique():
    idx = (df_unique.helicity == helicity)
    mean_iota = df_unique['mean_iota'][idx]
    aspect_ratio = df_unique['aspect_ratio'][idx]
    if helicity == 1:
        qs_label = 'QH'
    else:
        qs_label = 'QA'
    ax.scatter(mean_iota, aspect_ratio, c=colors[helicity], s=markersize, marker=markers[helicity], alpha=alpha, label='%s'%(qs_label), zorder=5)



""" Plot the out-of-sample devices """

# scatter some misc data points
ax.scatter([0.36], [4.5], color=colors[2], s=int(2*markersize), marker=markers[2], alpha=1.0, label=r'$n_{\text{fp}}=2$ QA', zorder=5)
ax.scatter([0.5], [18.5], color=colors[2], s=int(2*markersize), marker=markers[3], alpha=1.0, label=r'$n_{\text{fp}}=3$ QA', zorder=5)
ax.scatter([0.5], [9.0], color=colors[2], s=int(2*markersize), marker=markers[4], alpha=1.0, label=r'$n_{\text{fp}}=3$ QH', zorder=5)
ax.scatter([1.4], [11.0], color=colors[2], s=int(2*markersize), marker=markers[5], alpha=1.0, label=r'$n_{\text{fp}}=4$ QH', zorder=5)
ax.scatter([2.5], [17.0], color=colors[2], s=int(2*markersize), marker=markers[6], alpha=1.0, label=r'$n_{\text{fp}}=5$ QH', zorder=5)
ax.scatter([2.0], [14.0], color=colors[2], s=int(2*markersize), marker=markers[7], alpha=1.0, label=r'$n_{\text{fp}}=6$ QH', zorder=5)
ax.scatter([3.7], [11.0], color=colors[2], s=int(2*markersize), marker=markers[8], alpha=1.0, label=r'$n_{\text{fp}}=7$ QH', zorder=5)
ax.scatter([2.5], [22.0], color=colors[2], s=int(2*markersize), marker=markers[9], alpha=1.0, label=r'$n_{\text{fp}}=8$ QH', zorder=5)
xlim = ax.get_xlim()
ylim = ax.get_ylim()


""" 
Plot an image of a device
serial0000952 (iota = 0.1, aspect = 20, nfp2, QA)
https://quasr.flatironinstitute.org/model/0000952
"""
# img = plt.imread("./viz/serial0000952.png")
# device_iota = 0.1
# device_aspect = 20
# x_left = -0.8
# x_right = 0.2
# y_bottom = 21
# y_top = 29
# ax.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# top left
img = plt.imread("./viz/serial0000952.png")
device_iota = 0.1
device_aspect = 20
x_left = -0.3
x_right = 0.5
y_bottom = 0.5
y_top = 1.0
ax2.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# draw a letter on the right plot
ax2.text(-0.3, 0.95, "A", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', weight='bold')

# draw a letter on the left plot
x_start_arrow = device_iota - 0.1
y_start_arrow = device_aspect + 1
x_end_arrow = device_iota -0.03
y_end_arrow = device_aspect +0.3
ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
            xy=(x_end_arrow, y_end_arrow),
            arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4), zorder=100)  
ax.text(x_start_arrow - 0.1, y_start_arrow + 1, "A", fontsize=14,
        verticalalignment='top', weight='bold', zorder=150)

# # draw an arrow 
# x_start_arrow = -0.1
# y_start_arrow = 22.3
# x_end_arrow = device_iota - 0.04
# y_end_arrow = device_aspect + 0.4
# ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
#             xy=(x_end_arrow, y_end_arrow),
#             arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4))

""" 
Plot an image of a device

serial2022065 (iota = 2.9, aspect = 11.99 QH)
# https://quasr.flatironinstitute.org/model/2022065
"""
# img = plt.imread("./viz/serial2593103.png")
# device_iota = 3.0
# device_aspect = 11.93
# x_left = 2.5
# x_right = 3.9
# y_bottom = 13
# y_top = 21
# ax.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# bottom right
img = plt.imread("./viz/serial2022065.png")
device_iota = 2.9
device_aspect = 11.99
x_left = 0.4
x_right = 1.7
# y_bottom = 0.5
# y_top = 1.0
y_bottom = -0.02
y_top = 0.45
ax2.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# draw letter on the right plot
ax2.text(0.55, 0.45, "D", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', weight='bold')

# draw a letter on the left plot
x_start_arrow = device_iota + 0.1
y_start_arrow = device_aspect + 1
x_end_arrow = device_iota - 0.04
y_end_arrow = device_aspect + 0.4
ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
            xy=(x_end_arrow, y_end_arrow),
            arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4))  
ax.text(x_start_arrow + 0.03, y_start_arrow + 0.7, "D", fontsize=14,
        verticalalignment='top', weight='bold', zorder=150)

# # # draw an arrow 
# # x_start_arrow = -0.1
# # y_start_arrow = 22.3
# # x_end_arrow = device_iota - 0.04
# # y_end_arrow = device_aspect + 0.4
# # ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
# #             xy=(x_end_arrow, y_end_arrow),
# #             arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4))   

""" 
Plot an image of a device
serial1328281 (iota = 1.2, aspect = 8 QH)
# https://quasr.flatironinstitute.org/model/1328281
"""
# img = plt.imread("./viz/serial1328281.png")
# device_iota = 1.2
# device_aspect = 8.0
# x_left = 0.7
# x_right = 1.9
# y_bottom = 5
# y_top = -2
# ax.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# bottom left
img = plt.imread("./viz/serial1328281.png")
device_iota = 1.2
device_aspect = 8.0
x_left = -0.5
x_right = 0.8
y_bottom = -0.0
y_top = 0.45
ax2.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# place a text box in upper left in axes coords
ax2.text(-0.3, 0.45, "C", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', weight='bold')

# draw a letter on the left plot
x_start_arrow = 1.3
y_start_arrow = 4.9
x_end_arrow = device_iota + 0.01
y_end_arrow = device_aspect - 0.35
ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
            xy=(x_end_arrow, y_end_arrow),
            arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4), zorder=100)  
ax.text(x_start_arrow - 0.04, y_start_arrow - 0.1, "C", fontsize=14,
        verticalalignment='top', weight='bold', zorder=150)

# # # draw an arrow 
# # x_start_arrow = -0.1
# # y_start_arrow = 22.3
# # x_end_arrow = device_iota - 0.04
# # y_end_arrow = device_aspect + 0.4
# # ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
# #             xy=(x_end_arrow, y_end_arrow),
# #             arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4))   

""" 
Plot an image of a device
serial0040380 (iota = 0.1, aspect = 4 QA)
# https://quasr.flatironinstitute.org/model/0040380
"""
# img = plt.imread("./viz/serial0040380.png")
# device_iota = 0.1
# device_aspect = 4.0
# x_left = -0.8
# x_right = 0.2
# y_bottom = 9
# y_top = 1
# ax.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# top right
img = plt.imread("./viz/serial0040380.png")
device_iota = 0.1
device_aspect = 4.0
x_left = 0.5
x_right = 1.5
# y_bottom = -0.01
# y_top = 0.4
y_bottom = 0.55
y_top = 0.9
ax2.imshow(img, extent=[x_left, x_right, y_bottom, y_top], zorder=100, aspect='auto', clip_on=False)

# place a text box in upper left in axes coords
ax2.text(0.55, 0.95, "B", transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', weight='bold')

# draw a letter on the left plot
x_start_arrow = device_iota - 0.1
y_start_arrow = device_aspect + 1
x_end_arrow = device_iota -0.03
y_end_arrow = device_aspect +0.3
ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
            xy=(x_end_arrow, y_end_arrow),
            arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4), zorder=100)  
ax.text(x_start_arrow - 0.1, y_start_arrow + 1, "B", fontsize=14,
        verticalalignment='top', weight='bold', zorder=150)

# # # draw an arrow 
# # x_start_arrow = -0.1
# # y_start_arrow = 22.3
# # x_end_arrow = device_iota - 0.04
# # y_end_arrow = device_aspect + 0.4
# # ax.annotate("", xytext=(x_start_arrow, y_start_arrow),
# #             xy=(x_end_arrow, y_end_arrow),
# #             arrowprops=dict(facecolor='black', width=0.1, headwidth=4, headlength=4))   

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

# prevent the axis from zooming in on the image
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('auto')


ax.set_xlabel(r'Mean Rotational Transform $\bar{\iota}$')
ax.set_ylabel('Aspect Ratio')
ax.legend(loc='upper right', fontsize=10, framealpha=1.0)
plt.savefig(outdir + 'quasr_data_range.pdf', bbox_inches='tight', format='pdf')
plt.show()