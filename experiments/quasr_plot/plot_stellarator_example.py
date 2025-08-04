import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from simsopt._core import load
from simsopt.field import coils_via_symmetries, BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier
from diffusion_for_fusion.fourier_interpolation import fourier_interp1d_regular_grid

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 13})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange']
markers = ['s','o',  'x', '^', '*', 'p', 'D', 'v', '>', '<',  'h']
outdir = "./viz/"
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

"""
Plot a stellarator from QUASR.
serial1328281 (iota = 1.2, aspect = 8 QH)
https://quasr.flatironinstitute.org/model/1328281
"""

# load the data
[surfaces, base_coils] = load("./data/serial1328281.json")

fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12, 4), ncols=3)

# compute the cross sections for each surface
nfp = surfaces[-1].nfp
phi_list = [0, 1 / nfp / 4, 2 / nfp / 4, 3 / nfp / 4]
for ii, phi in enumerate(phi_list):
    for jj, surface in enumerate(surfaces):
        xyz= surface.cross_section(phi)
        R = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        Z = xyz[:, 2]

        R = fourier_interp1d_regular_grid(R, 512)
        Z = fourier_interp1d_regular_grid(Z, 512)
        # plot the cross sections
        ax2.plot(R, Z, color=colors[ii], lw=2)

# plot the field strength on the outer surface
surface = surfaces[-1]
ntheta = 256
nphi = 257
surface_plot = SurfaceXYZTensorFourier(
    mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=surface.nfp,
    quadpoints_phi=np.linspace(0, 1/surface.nfp, nphi, endpoint=True),
    quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=True))
surface.unfix_all()
surface_plot.unfix_all()
surface_plot.x = surface.x
base_curves = [c.curve for c in base_coils]
base_currents = [c.current for c in base_coils]
coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
biotsavart = BiotSavart(coils)
xyz = surface_plot.gamma().reshape((-1, 3))
biotsavart.set_points(xyz)
modB = biotsavart.AbsB()
phi = surface_plot.quadpoints_phi
theta = surface_plot.quadpoints_theta
phi, theta = np.meshgrid(phi, theta, indexing='ij')
modB = modB.reshape(np.shape(phi))
levels = np.linspace(np.min(modB), np.max(modB), 18, endpoint=False)
ax3.contour(phi, theta, modB, cmap='viridis', levels=levels, lw=2)
ax3.set_xlabel(r"$\varphi / 2\pi n_{fp}$")
ax3.set_ylabel(r"$\theta / 2\pi$")
ax3.set_xticks(np.linspace(0, 1/surface.nfp, 3, endpoint=True), [0, 0.5, 1])
ax3.set_yticks(np.linspace(0, 1, 3, endpoint=True), [0, 0.5, 1])


# plot the image
img = plt.imread("./viz/serial1328281_with_coils.png")
ax1.imshow(img, 
           extent=[0.05, 0.95, -0.3, 1.3],
           zorder=100, aspect='auto', clip_on=True, interpolation='bilinear')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])


ax1.set_yticks([])
ax1.set_xticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.tight_layout()
plt.savefig(outdir + 'stellarator_example.pdf', bbox_inches='tight', format='pdf')
plt.show()