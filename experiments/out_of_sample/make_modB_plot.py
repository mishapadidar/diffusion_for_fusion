import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
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

Prior to running this script, run generate_modB_data.py to generate the |B| data.
Then manually create images of the corresponding stellarators in paraview using the .vts files.
"""


# load |B| data
data_filelist = glob.glob("./viz/modB_data_*.pickle")
image_filelist = glob.glob("./viz/surface_*.png")
data_filelist.sort()
image_filelist.sort()

# sort by nfp, helicity
def sort_key(filename):
    parts = filename.split('_')
    if 'pickle' in filename:
        nfp = int(parts[5])
        helicity = int(parts[7])
    elif 'png' in filename:
        nfp = int(parts[4])
        helicity = int(parts[6])
    return (nfp, helicity)
data_filelist.sort(key=sort_key)
image_filelist.sort(key=sort_key)

fig, ax = plt.subplots(figsize=(18, 6), nrows=2, ncols=len(data_filelist), sharey=False)

for ii, datafile in enumerate(data_filelist):
    ax1 = ax[0][ii]
    ax2 = ax[1][ii]

    # load the data
    data = pickle.load(open(datafile, 'rb'))
    
    # plot |B| contours
    modB = data['modB']
    theta = data['theta']
    phi = data['phi']
    nfp = data['nfp']

    levels = np.linspace(np.min(modB), np.max(modB), 12, endpoint=False)
    ax2.contour(phi, theta, modB, cmap='viridis', levels=levels)

    # ax2.set_xlabel(r"$\varphi / 2\pi n_{\text{fp}}$")
    # ax2.set_ylabel(r"$\theta / 2\pi$")
    # ax2.set_xticks(np.linspace(0, 2*np.pi/nfp, 3, endpoint=True), [0, 0.5, 1])
    # ax2.set_yticks(np.linspace(0, 2*np.pi, 3, endpoint=True), [0, 0.5, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # plot the image
    img = plt.imread(image_filelist[ii])
    ax1.imshow(img, 
            extent=[-0.1, 1.1, -0.0, 1.0],
            zorder=100, aspect='auto', clip_on=True, interpolation='bilinear')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xticks([])
    ax1.set_yticks([])

    title = r"$n_{\text{fp}}=%d$"%(nfp)
    if data['helicity'] == 1:
        title += " QH"
    else:
        title += " QA"

    ax1.set_title(title, fontsize=16)


plt.tight_layout()
plt.savefig(outdir + 'out_of_sample_contours.pdf', bbox_inches='tight', format='pdf')
plt.show()


