import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier, CurveXYZFourier, CurveLength
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from scipy.optimize import minimize


def evaluate_configuration(x, iota, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, G=1):
    """Evaluate a Boozer Surface configuration by fitting coils and computing metrics.

    Args:
        x (np array)): 1d array of degrees of freedom for a SurfaceXYZTensorFourier.
        iota (float): rotational transform.
        nfp (int): number of field periods.
        stellsym (bool, optional): True for stellarator symmetric configurations. Defaults to True.
        mpol (int): number of poloidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        ntor (int): number of toroidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        helicity (int, optional): quasisymmetry helicity n. Defaults to 0 for QA. Use 1 or -1 for QH.
        G (float, optional): Boozer's G. Defaults to 1.

    Returns:
        metrics (dict): dictionary of metrics
    """

    # build the surface
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 2*ntor+1, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 2*mpol+1, endpoint=False))
    surf.x = x

    R1 = surf.minor_radius() * 4.0
    ncoils = int(np.ceil(16 / (2*surf.nfp))) # 16 coils total
    biotsavart, ma = stage2(surf, ncoils=ncoils, R1=R1, order=10, coil_current=1.0, length_target=30.0, length_weight=1.0)

    # compute QS-error
    quasi = "QA" if helicity == 0 else "QH"
    qs_error = NonQuasiSymmetricRatio(surf, biotsavart, quasi=quasi).J()
    qs_error = np.sqrt(qs_error)

    # compute boozer residual
    residual = boozer_residual(surf, biotsavart, iota, G) # (nphi, ntheta)
    residual_mse = np.mean(residual**2) 

    metrics = {'qs_error': qs_error, 
               'aspect_ratio': surf.aspect_ratio(), 
               'boozer_residual_mse': residual_mse,
               'biotsavart': biotsavart,
               'ma': ma,
               }
        

    return metrics

def boozer_residual(surface, biotsavart, iota, G):
    """
    Compute the residual of the Boozer equation on the surface,
        r = ||G * B - |B|^2 * (dx/dphi + iota * dx/dtheta)||

    Returns:
        residual (array): The residual of the Boozer equation, shape (nphi, ntheta).
    """
    dx_by_dphi = surface.gammadash1() # (nphi, ntheta, 3)
    dx_by_dtheta = surface.gammadash2()
    tangent = dx_by_dphi + iota * dx_by_dtheta # (nphi, ntheta, 3)
    
    pts = surface.gamma()
    nphi, ntheta, _ = np.shape(pts)
    pts = pts.reshape((-1, 3)) # (nphi*ntheta, 3)

    biotsavart.set_points(pts)
    B = biotsavart.B().reshape((nphi, ntheta, 3)) # (nphi, ntheta, 3)
    modB_squared = np.sum(B**2, axis=-1, keepdims=True)  # (nphi, ntheta, 1)

    residual = np.sum((G * B - modB_squared * tangent)**2, axis=-1)
    return residual

class NonQuasiSymmetricRatio:
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasisymmetric and
    non-quasisymmetric components.  For quasi-axisymmetry, we compute

    .. math::
        B_{\text{QS}} &= \frac{\int_0^1 B \|\mathbf n\| ~d\varphi}{\int_0^1 \|\mathbf n\| ~d\varphi} \\
        B_{\text{non-QS}} &= B - B_{\text{QS}}

    where :math:`B = \| \mathbf B(\varphi,\theta) \|_2`.  
    For quasi-poloidal symmetry, an analagous formula is used, but the integral is computed in the :math:`\theta` direction.
    The objective computed by this penalty is

    .. math::
        J &= \frac{\int_{\Gamma_{s}} B_{\text{non-QS}}^2~dS}{\int_{\Gamma_{s}} B_{\text{QS}}^2~dS} \\

    When :math:`J` is zero, then there is perfect QS on the given boozer surface. The ratio of the QS and non-QS components
    of the field is returned to avoid dependence on the magnitude of the field strength.  Note that this penalty is computed
    on an auxilliary surface with quadrature points that are different from those on the input Boozer surface.  This is to allow
    for a spectrally accurate evaluation of the above integrals. Note that if boozer_surface.surface.stellsym == True, 
    computing this term on the half-period with shifted quadrature points is ~not~ equivalent to computing on the full-period 
    with unshifted points.  This is why we compute on an auxilliary surface with quadrature points on the full period.

    Args:
        in_surface (SurfaceXYZTensorFourier): Surface object that is the Boozer surface. Note that the surface angles must
            be boozer angles for the objective to be meaningful.
        biotsavart (BiotSavart): BiotSavart object that contains the coils to compute the field on the surface.
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.  
        quasi_poloidal: string that determines the type of quasisymmetry to compute. 'QA' for quasi-axisymmetry,
            'QP' for quasi-poloidal symmetry, and 'QH' for quasi-helical symmetry.
    """

    # def __init__(self, boozer_surface, bs, sDIM=20, quasi='QA'):
    def __init__(self, in_surface, biotsavart, sDIM=20, quasi='QA'):
        assert isinstance(in_surface, SurfaceXYZTensorFourier)
        assert (quasi == 'QA') or (quasi == 'QH') or (quasi == 'QP')

        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM+1, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas, dofs=in_surface.dofs)

        self.quasi=quasi
        def make_QA_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            return idx, jdx

        def make_QP_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            return jdx, idx

        def make_QH_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            idx, jdx = np.mod(idx-jdx, in_nphi), np.mod(idx+jdx, in_ntheta)

            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            idx, jdx = np.mod(idx-jdx, in_nphi), np.mod(idx+jdx, in_ntheta)
            return idx, jdx
        
        if quasi == 'QH':
            self.idx, self.jdx = make_QH_matrix(phis.size, thetas.size)
        elif quasi == 'QP':
            self.idx, self.jdx = make_QP_matrix(phis.size, thetas.size)
        else:
            self.idx, self.jdx = make_QA_matrix(phis.size, thetas.size)
        
        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = biotsavart

    def J(self):
        # if self.boozer_surface.need_to_run_code:
        #     res = self.boozer_surface.res
        #     res = self.boozer_surface.run_code(res['type'], res['iota'], G=res['G'])
        idx = self.idx
        jdx = self.jdx

        pts = self.surface.gamma()[idx, jdx, :]
        self.biotsavart.set_points(pts.reshape((-1, 3)))
        
        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()[idx, jdx, :]
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_QS = B_QS[None, :]
        B_nonQS = modB - B_QS
        _J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)
        return _J
    
def create_equally_spaced_curves_around_axis(ma, ncurves, stellsym, R1=0.5, order=6, numquadpoints=64):
    """Initialize circular curves on uniformly spaced around the
    magnetic axis.

    Curve i is initialized around the axis point r0(phi_i) as,
        r_i(theta) = r0(phi_i) + R1 * cos(theta) * n(phi_i) + R1 * sin(theta) * b(phi_i)
    where n, b are the normal and binormal vectors at r0(phi_i). The angles phi_i
    are selected so that the curves are uniformly spaced around the entire torus
    when completed via symmetries.

    Args:
        ma (Curve): Curve object representing the axis
        ncurves (float): number of coils
        stellsym (bool): If True, the curves are initialized using only the first half of the points
        R1 (float): minor radius of the curves
        order (int): number of Fourier modes for the curves.
        numquadpoints (int): Number of quadrature points for the curves.

    Returns:
        list: list of CurveXYZFourier objects.
    """
    xyz = ma.gamma()
    nphi = xyz.shape[0] # number of phi points
    (_,normal,binormal) = ma.frenet_frame()

    # determine coil centers and axis tangent
    if not stellsym:
        end = nphi + 1
    else:
        end = (nphi + 1) // 2
    padding = end / ncurves / 2
    phi_idx = np.linspace(padding, end - padding, ncurves, dtype=int)
    centers = xyz[phi_idx] # (ncurves, 3)
    normals = normal[phi_idx] # (ncurves, 3)
    binormals = binormal[phi_idx] # (ncurves, 3)

    curves = []
    for i_curve in range(ncurves):
        curve = CurveXYZFourier(numquadpoints, order)
        # center the curve
        curve.set('xc(0)', centers[i_curve, 0])
        curve.set('yc(0)', centers[i_curve, 1])
        curve.set('zc(0)', centers[i_curve, 2])
        # orient the curve
        curve.set('xc(1)', R1 * normals[i_curve, 0])
        curve.set('yc(1)', R1 * normals[i_curve, 1])
        curve.set('zc(1)', R1 * normals[i_curve, 2])
        curve.set('xs(1)', R1 * binormals[i_curve, 0])
        curve.set('ys(1)', R1 * binormals[i_curve, 1])
        curve.set('zs(1)', R1 * binormals[i_curve, 2])
        curves.append(curve)

    return curves

def stage2(surface, ncoils=4, R1=1, order=10, coil_current=1.0, length_target=30.0, length_weight=1.0, maxiter=2000, ftol=1e-12):
    """
    Fit coils to a Boozer surface using stage-2 optimization,
        min J = Jf + length_weight * QuadraticPenalty(Jls, length_target, 'max')
    where
        Jf is the normalized squared flux objective function, and
        Jls are the curve lengths of the coils.

    This function only works for stellsymetric configurations.

    Args:
        surface (SurfaceXYZTensorFourier): Boozer surface to fit coils to.
            It is assume that the quadrature points lie on 1/2 field period.
        ncoils (int, optional): Number of coils to fit. Defaults to 4.
        R1 (float, optional): Minor radius of the coils. Defaults to 1.
        order (int, optional): Fourier order of the coils. Defaults to 10.
        coil_current (float, optional): Current in each coil. Defaults to 1.0.
        length_target (float, optional): Target length for each coil. Defaults to 30.0.
        length_weight (float, optional): Weight for the coil length penalty. Defaults to 1.0.

    Returns:
        BiotSavart: BiotSavart object containing the fitted coils.
    """
    # guess a magnetic axis (on first half field period)
    ma = CurveXYZFourier(quadpoints=surface.quadpoints_phi, order=surface.ntor)
    xyz = surface.gamma()
    xyz0 = np.mean(xyz, axis=(1)) # (nphi, 3)
    ma.least_squares_fit(xyz0)

    # place coils around first half period
    stellsym = surface.stellsym
    base_curves = create_equally_spaced_curves_around_axis(ma, ncoils, stellsym=False, R1=R1, order=order, numquadpoints=64)

    # generate remaining coils
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, stellsym=stellsym)
    biotsavart = BiotSavart(coils)

    base_currents[0].fix_all()

    # Define the individual terms objective function:
    Jf = SquaredFlux(surface, biotsavart, definition="normalized")
    Jls = [CurveLength(c) for c in base_curves]

    # Form the total objective function.
    objective = Jf + length_weight * QuadraticPenalty(sum(Jls), length_target, "max")

    def fun(dofs):
        objective.x = dofs
        return objective.J(), objective.dJ()

    res = minimize(fun, objective.x, jac=True, method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxcor':300, 'ftol':ftol}, tol=1e-15)
    objective.x = res.x

    print(res)

    return biotsavart, ma

def test_stage2():
    """Test stage2 by loading a data point from QUASR."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # parameters
    mpol = ntor = 10 # always 10 for quasr
    stellsym = True
    
    # which data point
    idx_data = 1203
    idx_data = 23525

    # load the data set
    Y_init = pd.read_pickle('../data/QUASR.pkl') # y-values
    X_init = np.load('../data/dofs.npy') # x-values
    Y_init = Y_init.reset_index(drop=True)
    conditions = ["qs_error", 'iota_profile', "aspect_ratio", "nfp", "helicity", "currents", 'nc_per_hp']
    Y = Y_init[conditions]
    X = X_init

    # pick a data point
    x = X[idx_data]
    nfp = Y.nfp[idx_data]
    I_P = sum(np.abs(Y.currents[0])) * nfp * 2 # total current through hole
    G = (4 * np.pi * 1e-7) * I_P
    iota = Y.iota_profile[idx_data][-1] # rotational transform

    # build the surface on half field period
    quadpoints_phi = np.linspace(0, 1 / (2 * nfp), 2*ntor + 1, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf.x = np.copy(x)

    # run stage2
    R1 = surf.minor_radius() * 4.0
    ncoils = int(np.ceil(16 / (2*surf.nfp)))
    print(ncoils)
    import time
    t0 = time.time()
    biotsavart, ma = stage2(surf, ncoils=ncoils, R1=R1, order=10, coil_current=1.0, length_target=30.0, length_weight=1.0)
    t1 = time.time()
    print("Stage 2 optimization took", t1 - t0, "seconds")

    # plot the coils
    from simsopt.geo import curves_to_vtk
    import os
    curves = [c.curve for c in biotsavart.coils]
    outdir = "./output/"
    os.makedirs(outdir, exist_ok=True)
    curves_to_vtk(curves, filename=outdir + 'coils')
    surf.to_vtk(filename=outdir+'surface')
    curves_to_vtk([ma], filename=outdir+'magnetic_axis')



def test_evaluate_configuration():
    """Test the evaluate_configuration function by loading a data point from QUASR."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # parameters
    mpol = ntor = 10 # always 10 for quasr
    stellsym = True
    
    # which data point
    idx_data = 1203

    # load the data set
    Y_init = pd.read_pickle('../data/QUASR.pkl') # y-values
    X_init = np.load('../data/dofs.npy') # x-values
    Y_init = Y_init.reset_index(drop=True)
    conditions = ["qs_error", 'iota_profile', "aspect_ratio", "nfp", "helicity", "currents", 'nc_per_hp']
    Y = Y_init[conditions]
    X = X_init

    print(np.sqrt(Y.qs_error[idx_data]), Y.iota_profile[idx_data][-1], Y.aspect_ratio[idx_data], Y.nfp[idx_data], Y.helicity[idx_data])

    # pick a data point
    x = X[idx_data]
    nfp = Y.nfp[idx_data]
    I_P = sum(np.abs(Y.currents[0])) * nfp * 2 # total current through hole
    G = (4 * np.pi * 1e-7) * I_P
    iota = Y.iota_profile[idx_data][-1] # rotational transform

    # evaluate the configuration
    metrics = evaluate_configuration(x, iota, nfp, stellsym=stellsym, mpol=mpol, ntor=ntor, helicity=Y.helicity[idx_data], G=G)
    print('metrics', metrics)

    # build the surface
    quadpoints_phi = np.linspace(0, 1 / (nfp), 2*ntor + 1, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf.x = np.copy(x)

    # contour plot
    xyz = surf.gamma().reshape((-1, 3))
    biotsavart = metrics['biotsavart']
    biotsavart.set_points(xyz)
    modB = biotsavart.AbsB().reshape((surf.quadpoints_phi.size, surf.quadpoints_theta.size))
    X, Y = np.meshgrid(surf.quadpoints_phi, surf.quadpoints_theta, indexing='ij')
    plt.contourf(X, Y, modB, cmap='viridis', levels=50)
    plt.xlabel('$\phi$')
    plt.ylabel(r'$\theta$')
    plt.show()


if __name__ == "__main__":
    # test_stage2()
    test_evaluate_configuration()