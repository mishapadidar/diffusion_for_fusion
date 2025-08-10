import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from diffusion_for_fusion.evaluate_configuration_vmec import evaluate_configuration as evaluate_configuration_vmec
from simsopt.mhd import Boozer
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
from simsopt.geo import SurfaceXYZTensorFourier
import pickle


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
This script generates the |B| data for a configuration sampled by the diffusion model.
    - Chooses a typical configuration and 
    - Compute |B| contours in Boozer coordinates
    - write a vtk file of the surface with |B|.

Prior to running this script, run evaluate_model.py to generate the diffusion samples and metrics.
"""


# load model evaluations
tag_list = ["iota_0.36_nfp_2_helicity_1_aspect_ratio_4.5",
            "iota_0.5_nfp_3_helicity_1_aspect_ratio_9.0",
            ]

ntheta = 256
nphi = 257


df_list = [pd.read_csv("./output/diffusion_metrics_" + tag + ".csv") for tag in tag_list]
X_list = [np.load("./output/diffusion_samples_" + tag + ".npy") for tag in tag_list]


config_list = []
for ii, df in enumerate(df_list):
    print("")
    # find index of configuration with median qs_error
    median_qs_error = df['sqrt_non_qs_error'].min()
    # idx_best = df['sqrt_non_qs_error'].sub(median_qs_error).abs().idxmin()
    idx_best = df['sqrt_non_qs_error'].idxmin()
    nfp = round(df.iloc[idx_best]['nfp'])
    helicity = round(df.iloc[idx_best]['helicity'])
    print(np.shape(X_list[ii]))
    print(len(df))
    xx = X_list[ii][idx_best]

    print(idx_best, median_qs_error)
    print("nfp", nfp, "helicity", helicity)


    # evaluate the configuration
    metrics, vmec = evaluate_configuration_vmec(xx, nfp, helicity=helicity, vmec_input="../../diffusion_for_fusion/input.nfp4_template")
    print(f"success={metrics['success']}, sqrt_non_qs_error={metrics['sqrt_non_qs_error']}, aspect_ratio={metrics['aspect_ratio']}, mean_iota={metrics['mean_iota']}")

    # get |B| in boozer coordinates
    booz = Boozer(vmec)
    booz.register([1.0]) # register surface
    booz.run()
    bx = booz.bx
    # discretize boozer angles
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d = np.linspace(0, 2 * np.pi / bx.nfp, nphi)
    phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')
    # index of flux surface
    js = 0
    # reconstruct |B| and sqrtg
    modB = np.zeros(np.shape(phi))
    for jmn in range(len(bx.xm_b)):
        m = bx.xm_b[jmn]
        n = bx.xn_b[jmn]
        angle = m * theta - n * phi
        modB += bx.bmnc_b[jmn, js] * np.cos(angle)
        if bx.asym:
            modB += bx.bmns_b[jmn, js] * np.sin(angle)

    # save the |B| data for plotting contours
    data = {
        'modB': modB,
        'theta': theta,
        'phi': phi,
        'nfp': bx.nfp,
    }
    outfilename = outdir + f'modB_data_{tag_list[ii]}.pickle'
    with open(outfilename, 'wb') as f:
        pickle.dump(data, f)
    print("Saved modB data to", outfilename)
    
    # write vtk file
    surface = vmec.boundary
    surface.unfix_all()
    quadpoints_phi=np.linspace(0, 1, nphi, endpoint=True)
    quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=True)
    surface_plot = SurfaceXYZTensorFourier(
        mpol=10, ntor=10, stellsym=True, nfp=nfp, quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)
    surface_plot.unfix_all()
    surface_plot.x = surface.x
    data = vmec_compute_geometry(vmec, s = 1.0, theta=2*np.pi*quadpoints_theta, phi=2*np.pi*quadpoints_phi)
    modB_surface = data.modB[0].T # (nphi, ntheta)
    pointData = {"modB": modB_surface[:, :, None]}
    outfilename = outdir + f'surface_{tag_list[ii]}'
    surface_plot.to_vtk(outfilename, extra_data=pointData)
    print("Saved vtk to", outfilename)









