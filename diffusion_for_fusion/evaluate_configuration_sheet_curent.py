import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from diffusion_for_fusion.sheet_current import SheetCurrent

def evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, M=10, N=10, G=1, ntheta=21, nphi=21, extend_factor=0.2):
    """Evaluate a surface configuration with the SheetCurrent vacuum solver.

    Args:
        x (np array)): 1d array of degrees of freedom for a SurfaceXYZTensorFourier.
        nfp (int): number of field periods.
        stellsym (bool, optional): True for stellarator symmetric configurations. Defaults to True.
        mpol (int): number of poloidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        ntor (int): number of toroidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        helicity (int, optional): quasisymmetry helicity n. Defaults to 0 for QA. Use 1 or -1 for QH.
        M (int, optional): number of poloidal modes for the sheet current. Defaults to 10.
        N (int, optional): number of toroidal modes for the sheet current. Defaults to 10.
        G (float, optional): Boozer's G. Defaults to 1.
        ntheta (int, optional): minimum number of surface (not winding surface) quadrature points in theta direction. 
            The actual ntheta will be max(2*mpol +1, ntheta). Defaults to 31.
        nphi (int, optional): minimum number of surface (not winding surface) quadrature points in phi direction.
            The actual nphi will be max(2*mpol +1, nphi). Defaults to 31.
        extend_factor (float, optional): factor by which to extend the winding surface in the normal direction. The total extension
            distance extended is a multiple of the major radius. Defaults to 0.2.

    Returns:
        tuple: (metrics, sheet_current)
        metrics (dict): dictionary of metrics
        sheet_current (SheetCurrent): SheetCurrent object with the fitted current.
    """

    # build the surface
    ntheta = max(2*mpol+1, ntheta)
    nphi = max(2*ntor+1, nphi)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    surf.x = x

    # surf = SurfaceXYZTensorFourier(
    #     mpol=10, ntor=10, stellsym=True, nfp=nfp,
    #     quadpoints_phi=np.linspace(0, 1 / nfp, 31, endpoint=False),
    #     quadpoints_theta=np.linspace(0, 1, 31, endpoint=False))

    assert 2*M+1 <= ntheta and N <= 2*nphi + 1, "M and N must be less than or equal to ntheta and nphi respectively."

    # build winding surface
    surf_winding = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False), # one field period
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    surf_winding.x = np.copy(x)  
    orientation = compute_orientation(surf_winding)
    # dist_extend = surf_winding.minor_radius()*(orientation * extend_factor)
    dist_extend = surf_winding.major_radius() * (orientation * extend_factor)
    surf_winding.extend_via_normal(dist_extend)

    # fit the sheet current
    current = SheetCurrent(surf_winding, G, M, N)
    current.fit(surf)

    # compute iota using the least-squares estimate
    iota = compute_iota(surf, current)

    # compute QS-error
    quasi = "QA" if helicity == 0 else "QH"
    qs_error = NonQuasiSymmetricRatio(surf, current, quasi=quasi).J()
    qs_error = np.sqrt(qs_error)

    # compute boozer residual
    _, residual_mse = boozer_residual(surf, current, iota)
    
    metrics = {'sqrt_qs_error': qs_error, 'iota': iota, 'aspect_ratio': surf.aspect_ratio(), 'boozer_residual_mse': residual_mse}
    # metrics = np.array([qs_error, iota, surf.aspect_ratio(), residual_mse])

    return metrics, current


def compute_iota(surf, current):
    """Compute the iota value from the surface and current by minimizing the boozer residual.
    
    NOTE: The iota value produced by this function is only accurate if the boozer residual is 
    zero, i.e. the angles are indeed Boozer angles.

        r(theta, phi; iota) = G * B - mod(B)^2 * (dSigma/dvarphi + iota * dSigma/theta)
    as a least squares problem
        min_iota int || r(theta, phi; iota) ||^2 dA
    
    To simplify, define
        y =  mod(B)^2 * dSigma/dvarphi - G * B 
        x = mod(B)^2 * dSigma/theta
    then we solve,
        min_iota int || iota * x  - y||^2 dA
    and the solution is given by
        iota = int (y * x) dA / int(x * x) dA

    Args:
        surf (SurfaceXYZTensorFourier): Surface object.
        current (SheetCurrent): SheetCurrent object.

    Returns:
        float: iota value.
    """
    dtheta = np.diff(surf.quadpoints_theta)[0]
    dphi = np.diff(surf.quadpoints_phi)[0]
    normal = surf.normal().reshape(-1, 3) # (n, 3)
    dA = np.linalg.norm(normal, axis=-1) * (dtheta * dphi) # (n)

    dSigma_dphi = surf.gammadash1().reshape(-1, 3) # (n, 3)
    dSigma_dtheta = surf.gammadash2().reshape(-1, 3) # (n, 3)

    B = current.B(surf.gamma().reshape((-1, 3))) # (n, 3)
    modB = np.linalg.norm(B, axis=-1) # (n)

    y = (modB**2)[:, None] * dSigma_dphi - current.G * B  # (n, 3)
    x = (modB**2)[:, None] * dSigma_dtheta # (n, 3)

    top = np.sum(np.sum(y * x, axis=-1) * dA) # (3,)
    bottom = np.sum(np.sum(x**2, axis=-1) * dA) # (3,)

    iota = top / bottom

    return iota


def boozer_residual(surf, current, iota):
    """Compute the boozer residual for a given surface and current.

    Args:
        surf (SurfaceXYZTensorFourier): Surface object.
        current (SheetCurrent): SheetCurrent object.
        iota (float): rotational transform.

    Returns:
        np.ndarray: boozer residual vector.
        float: sqrt(mean-squared residual) boozer residual.

    """
    dtheta = np.diff(surf.quadpoints_theta)[0]
    dphi = np.diff(surf.quadpoints_phi)[0]
    normal = surf.normal().reshape(-1, 3) # (n, 3)
    dA = np.linalg.norm(normal, axis=-1) * (dtheta * dphi) # (n)

    dSigma_dphi = surf.gammadash1().reshape(-1, 3) # (n, 3)
    dSigma_dtheta = surf.gammadash2().reshape(-1, 3) # (n, 3)

    B = current.B(surf.gamma().reshape((-1, 3))) # (n, 3)
    modB = np.linalg.norm(B, axis=-1) # (n)

    # compute the boozer residual
    residual = current.G * B - (modB**2)[:, None] * (dSigma_dphi + iota * dSigma_dtheta)

    residual = np.linalg.norm(residual, axis=-1)  # (n,)
    total = np.sqrt(np.mean(np.sum(residual**2,axis=-1) * dA)) # (scalar)

    return residual, total

def compute_orientation(surf):
    """Compute the orientation of a surface. 

    We use the following observation to determine the orientation:
        volume  = int dV 
                = int div(F) dV where F = [x, y, z] / 3
                = int F . n dS
    If n is outward facing then the integral will be positive, otherwise 
    the integral will be negative.


    Args:
        surf (Surface): Simsopt Surface
    Returns:
        float: 1 if outward oriented, -1 if inward oriented.
    """

    F = surf.gamma() / 3
    n = surf.normal()
    vol = np.mean(np.sum(F * n, axis=-1))  # (scalar)
    if vol > 0:
        return 1.0
    elif vol < 0:
        return -1.0
    else:
        raise ValueError("Surface orientation is ambiguous, volume is zero.")


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
        sheet_current (SheetCurrent): SheetCurrent object that contains the Biot-Savart law for the surface.
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.  
        quasi_poloidal: string that determines the type of quasisymmetry to compute. 'QA' for quasi-axisymmetry,
            'QP' for quasi-poloidal symmetry, and 'QH' for quasi-helical symmetry.
    """

    # def __init__(self, boozer_surface, bs, sDIM=20, quasi='QA'):
    def __init__(self, in_surface, sheet_current, sDIM=20, quasi='QA'):
        # only SurfaceXYZTensorFourier for now
        # assert type(boozer_surface.surface) is SurfaceXYZTensorFourier 
        assert isinstance(in_surface, SurfaceXYZTensorFourier)
        assert (quasi == 'QA') or (quasi == 'QH') or (quasi == 'QP')

        # in_surface = boozer_surface.surface
        # self.boozer_surface = boozer_surface

        surface = in_surface
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
        self.sheet_current = sheet_current

    def J(self):
        # if self.boozer_surface.need_to_run_code:
        #     res = self.boozer_surface.res
        #     res = self.boozer_surface.run_code(res['type'], res['iota'], G=res['G'])
        idx = self.idx
        jdx = self.jdx

        pts = self.surface.gamma()[idx, jdx, :]
        # self.biotsavart.set_points(pts.reshape((-1, 3)))
        
        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        # B = self.biotsavart.B()
        B = self.sheet_current.B(pts.reshape((-1, 3)))
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()[idx, jdx, :]
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_QS = B_QS[None, :]
        B_nonQS = modB - B_QS
        _J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)
        return _J

def test_evaluate_configuration():
    """Test the evaluate_configuration function by loading a data point from QUASR."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # parameters
    M = N = 10
    extend_factor = 0.1
    nphi = ntheta = 31
    # which data point
    idx_data = 35
    # idx_data = 1203
    # idx_data = 368248
    # idx_data = 212450
    # idx_data = 110825
    # idx_data = 239324

    # load the data set
    Y_init = pd.read_pickle('../data/QUASR.pkl') # y-values
    X_init = np.load('../data/dofs.npy') # x-values
    Y_init = Y_init.reset_index(drop=True)
    # print(Y_init.columns)
    conditions = ["qs_error", 'iota_profile', "aspect_ratio", "nfp", "helicity", "currents", 'nc_per_hp']
    Y = Y_init[conditions]
    X = X_init

    # pick a data point
    x = X[idx_data]
    nfp = Y.nfp[idx_data]
    I_P = sum(np.abs(Y.currents[0])) * nfp * 2 # total current through hole
    G = (4 * np.pi * 1e-7) * I_P
    stellsym = True
    mpol = ntor = 10 # for 661 degrees of freedom

    # build the surface
    ntheta = max(2*mpol+1, ntheta)
    nphi = max(2*ntor+1, nphi)
    quadpoints_phi = np.linspace(0, 1 / (nfp), nphi, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=False)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf.x = np.copy(x)

    # evaluate the configuration
    metrics, current = evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=Y.helicity[idx_data], M=M, N=N, G=G, ntheta=ntheta, nphi=nphi, extend_factor=extend_factor)
    print('metrics', metrics)

    # Plot the surface of xyz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xyz = surf.gamma()
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], alpha=0.5, label='surf')
    xyz = current.surface.gamma()
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], alpha=0.5, label='winding surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

    # contour plot
    xyz = surf.gamma()
    B = current.B(xyz.reshape((-1, 3))) # (n, 3)
    modB = np.linalg.norm(B, axis=-1) # (n)
    modB = modB.reshape(xyz.shape[:-1])
    X, Y = np.meshgrid(surf.quadpoints_phi, surf.quadpoints_theta, indexing='ij')
    plt.contourf(X, Y, modB, cmap='viridis', levels=50)
    plt.xlabel('$\phi$')
    plt.ylabel(r'$\theta$')
    plt.show()

    # Plot the surface of xyz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xyz = surf.gamma()
    B = current.B(xyz.reshape((-1, 3))) # (n, 3)
    modB = np.linalg.norm(B, axis=-1) # (n)
    modB = modB.reshape(xyz.shape[:-1])
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], facecolors=plt.cm.plasma((modB -np.min(modB))/ (np.max(modB) -np.min(modB))), alpha=1.0, label='surf')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    test_evaluate_configuration()