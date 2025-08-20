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




ntheta = 511
nphi = 512


# load model evaluations
df_filelist = glob.glob("./output/diffusion_metrics_*.csv")
X_filelist = glob.glob("./output/diffusion_samples_*.npy")
df_filelist.sort()
X_filelist.sort()
df_list = [pd.read_csv(ff) for ff in df_filelist]
X_list = [np.load(ff) for ff in X_filelist]


config_list = []
for ii, df in enumerate(df_list):
    print("")

    df['mean_iota_error'] = 100 * (df['mean_iota'] - df['mean_iota_condition']).abs() / df['mean_iota_condition']
    df['aspect_ratio_error'] = 100 * (df['aspect_ratio'] - df['aspect_ratio_condition']).abs() / df['aspect_ratio_condition']

    idx_downsample = (df['mean_iota_error'] < 5) & (df['aspect_ratio_error'] < 5)
    XX = X_list[ii][idx_downsample]
    df = df[idx_downsample].reset_index(drop=True)
    print("Number of configurations after downsampling:", len(df))

    idx_best = df['sqrt_qs_error_2term'].idxmin()
    nfp = round(df.iloc[idx_best]['nfp'])
    helicity = round(df.iloc[idx_best]['helicity'])
    mean_iota_condition = df.iloc[idx_best]['mean_iota_condition']
    aspect_ratio_condition = df.iloc[idx_best]['aspect_ratio_condition']
    xx = XX[idx_best]

    print("selected configuration", ii, "with qs error", df['sqrt_non_qs_error'].min(), "nfp", nfp, "helicity", helicity)


    # evaluate the configuration
    metrics, vmec = evaluate_configuration_vmec(xx, nfp, helicity=helicity, vmec_input="../../diffusion_for_fusion/input.nfp4_template")
    print(f"success={metrics['success']}, sqrt_non_qs_error={metrics['sqrt_non_qs_error']}, aspect_ratio={metrics['aspect_ratio']}, mean_iota={metrics['mean_iota']}")
    iota_err = 100 * abs(metrics['mean_iota'] - mean_iota_condition) / mean_iota_condition
    print("c_iota [%]", iota_err)
    aspect_err = 100 * abs(metrics['aspect_ratio'] - aspect_ratio_condition) / aspect_ratio_condition
    print("c_aspect [%]", aspect_err)

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
        'helicity': helicity,
        'aspect_ratio': metrics['aspect_ratio'],
        'mean_iota': metrics['mean_iota'],
        'sqrt_non_qs_error': metrics['sqrt_non_qs_error'],
    }
    tag = df_filelist[ii].split("metrics_")[-1].split(".csv")[0]
    outfilename = outdir + f'modB_data_{tag}.pickle'
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
    outfilename = outdir + f'surface_{tag}'
    surface_plot.to_vtk(outfilename, extra_data=pointData)
    print("Saved vtk to", outfilename)









