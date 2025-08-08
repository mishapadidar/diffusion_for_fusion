import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from diffusion_for_fusion.evaluate_configuration_vmec import evaluate_configuration as evaluate_configuration_vmec

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
Plot |B| data to show quasi-symmetry alongside an image of the stellarator
"""


# TODO: load |B| data
tag_list = ["iota_0.36_nfp_2_helicity_1_aspect_ratio_4.5",
            "iota_0.5_nfp_3_helicity_1_aspect_ratio_9.0"]

fig, ax = plt.subplots(figsize=(8, 6), nrows=2, ncols=len(tag_list))

for ii, tag in enumerate(tag_list):
    ax1 = ax[0][ii]
    ax2 = ax[1][ii]

    # TODO: load the data
    data_file = ...
    data = ...
    
    # TODO: Plot |B| contours
    modB = data['modB']
    theta = data['theta']
    phi = data['phi']
    nfp = data['nfp']

    levels = np.linspace(np.min(modB), np.max(modB), 18, endpoint=False)
    ax1.contour(phi, theta, modB, cmap='viridis', levels=levels, lw=2)

    ax1.set_xlabel(r"$\varphi / 2\pi n_{\text{fp}}$")
    ax1.set_ylabel(r"$\theta / 2\pi$")
    ax1.set_xticks(np.linspace(0, 1/nfp, 3, endpoint=True), [0, 0.5, 1])
    ax1.set_yticks(np.linspace(0, 1, 3, endpoint=True), [0, 0.5, 1])


    # TODO: plot the image
    # image_name = "./viz/....png"
    # img = plt.imread()
    # ax2.imshow(img, 
    #         extent=[0.05, 0.95, -0.3, 1.3],
    #         zorder=100, aspect='auto', clip_on=True, interpolation='bilinear')
    # ax2.set_xlim([0, 1])
    # ax2.set_ylim([0, 1])

plt.tight_layout()
# plt.savefig(outdir + 'stellarator_example.pdf', bbox_inches='tight', format='pdf')
plt.show()


