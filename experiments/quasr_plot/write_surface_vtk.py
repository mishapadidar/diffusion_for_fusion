import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier, curves_to_vtk
from simsopt.field import coils_via_symmetries, BiotSavart
from simsopt._core import load
import pandas as pd

"""
Write vtk files for a handful of (iota, aspect ratio) values.

You will have to manually plot these in Paraview.
"""
filelist = ["serial0000952", # https://quasr.flatironinstitute.org/model/0000952
            # "serial2557132", # https://quasr.flatironinstitute.org/model/2557132
            "serial2593103", # https://quasr.flatironinstitute.org/model/2593103
            "serial0040380", # https://quasr.flatironinstitute.org/model/0040380
            "serial1328281", # https://quasr.flatironinstitute.org/model/1328281
            ]

# load quasr data
for device_tag in filelist:
    # load the data
    [surfaces, base_coils] = load("./data/" + device_tag + '.json')
    surface = surfaces[-1]
    base_curves = [c.curve for c in base_coils]
    base_currents = [c.current for c in base_coils]

    # make a biot-savart object
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, surface.stellsym)
    biotsavart = BiotSavart(coils)

    # make the surface
    ntheta = 256
    nphi = 257
    surface_plot = SurfaceXYZTensorFourier(
        mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=surface.nfp,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=True),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=True))
    surface.unfix_all()
    surface_plot.unfix_all()
    surface_plot.x = surface.x

    # evaluate modB
    xyz = surface_plot.gamma().reshape((-1, 3))
    biotsavart.set_points(xyz)
    modB = biotsavart.AbsB()
    modB = modB.reshape((nphi, ntheta))

    # write the vtk file
    surface_plot.to_vtk("./viz/" + device_tag, extra_data = {'modB': modB.T.flatten()})
    print(f"Wrote vtk file for {device_tag} to ./viz/{device_tag}.vtk")
    print("Aspect Ratio:", surface_plot.aspect_ratio(), "nfp", surface_plot.nfp)

    # write a coil vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, "./viz/" + device_tag + "_coils")