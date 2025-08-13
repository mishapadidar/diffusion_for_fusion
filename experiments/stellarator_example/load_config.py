#!/usr/bin/env python
from simsopt._core import load, save
from simsopt.geo import SurfaceXYZTensorFourier, CurveRZFourier, curves_to_vtk, CurveXYZFourier
from simsopt.field import BiotSavart
import numpy as np

[surfaces, axis, coils] = load(f'serial0010198.json')

surface = SurfaceXYZTensorFourier(mpol=10, ntor=10, nfp=surfaces[-1].nfp, stellsym=True, quadpoints_phi=np.linspace(0, 0.5, 400), quadpoints_theta=np.linspace(0, 1, 100))
for idx,s in enumerate(surfaces):
    surface.x = s.x
    surface.to_vtk(f"symmetries{idx}")

bs = BiotSavart(coils)
surface = SurfaceXYZTensorFourier(mpol=10, ntor=10, nfp=surfaces[-1].nfp, stellsym=True, quadpoints_phi=np.linspace(0, 1, 400), quadpoints_theta=np.linspace(0, 1, 100))
for idx,s in enumerate(surfaces):
    surface.x = s.x
    bs.set_points(surface.gamma().reshape((-1, 3)))
    nphi, ntheta, _ = surface.gamma().shape
    B = bs.AbsB().reshape((1, nphi, ntheta))
    surface.to_vtk(f"symmetries_full{idx}", extra_data={'modB':B})

# make a field line on the outermost surface, this iota is not the correct one though
surface = SurfaceXYZTensorFourier(mpol=10, ntor=10, nfp=surfaces[-1].nfp, stellsym=True, quadpoints_phi=np.linspace(0, 0.5, 400), quadpoints_theta=np.linspace(0, 1, 400))
surface.x = surfaces[-1].x
idx0 = 200
idx = ((np.arange(400)+idx0)%400, np.arange(400))
xyz = surface.gamma()[idx]
np.savetxt('xyz.txt', xyz, header='x,y,z', delimiter=',')

curves = []
for c in coils[:2]:
    curve = CurveXYZFourier(np.linspace(0, 1, 160), c.curve.order)
    curve.x = c.curve.x
    curves.append(curve)
curves_to_vtk(curves, 'coils')

ma = CurveRZFourier(np.linspace(0, 1, 100), axis.order, 2, True)
ma.x = axis.x
curves_to_vtk([ma], 'axis')
