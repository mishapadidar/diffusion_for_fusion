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
Choose a typical configuration and 
- Plot |B| contours in Boozer coordinates
- write a vtk file of the surface with |B|.
"""


# load model evaluations
tag_list = ["iota_0.36_nfp_2_helicity_1_aspect_ratio_4.5",
            "iota_0.5_nfp_3_helicity_1_aspect_ratio_9.0"]
df_list = [pd.read_csv("./output/diffusion_metrics_" + tag + ".csv") for tag in tag_list]
X_list = [np.load("./output/diffusion_samples_" + tag + ".npy") for tag in tag_list]

# choose configuration with median qs_error
config_list = []
for ii, df in enumerate(df_list):    
    # find index of configuration with median qs_error
    median_qs_error = df['sqrt_non_qs_error'].median()
    idx_best = df['sqrt_non_qs_error'].sub(median_qs_error).abs().reset_index(drop=True).idxmin()

    xx = X_list[ii][idx_best]
    # print(f"Median qs error: {median_qs_error}, selected config index: {config_idx[-1]}")
    # print(df.iloc[idx_best])

    # TODO: evaluate the configuration
    metrics, _ = evaluate_configuration_vmec(xx, round(nfp_condition), helicity=round(helicity_condition), vmec_input="../../diffusion_for_fusion/input.nfp4_template")

    # TODO: get |B| on surface with boozxform

    # TODO: write vtk file

    # TODO: save the |B| data for plotting contours





