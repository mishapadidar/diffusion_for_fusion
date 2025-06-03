import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from sheet_current import SheetCurrent
from diffusion_for_fusion.utils import compute_iota, NonQuasiSymmetricRatio, compute_orientation, boozer_residual

def evaluate_configuration(x, nfp, mpol, ntor, helicity_n=0, vmec_input=None, stellsym=True,
                           plot=False):
    """Evaluate a surface configuration with VMEC.

    Args:
        x (np array)): 1d array of degrees of freedom for a SurfaceXYZTensorFourier.
        nfp (int): number of field periods.
        mpol (int): number of poloidal modes.
        ntor (int): number of toroidal modes.
        helicity_n (int, optional): quasisymmetry helicity n. Defaults to 0 for QA.
        vmec_input (str, optional): input file for vmec. Defaults to None.
        quadpoints_phi (int, optional): int or array of surface quadrature points in phi direction. Defaults to 31.
        quadpoints_theta (int, optional): int or array of surface quadrature points in theta direction. Defaults to 31.
        stellsym (bool, optional): True for stellarator symmetric configurations. Defaults to True.

    Returns:
        tuple: (quasisymmetry error, iota)
    """

    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp)
    surf.x = x
    vmec = Vmec(vmec_input, verbose=False)
    vmec.boundary = surf

    if plot:
        surf.plot(show=True)

    is_success = True

    try:
        qs = QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1,1,10,endpoint=False), helicity_n=helicity_n)
    except:
        is_success = False
        
    try:
        res = qs.compute()
    except:
        is_success = False
    
    dim_out=3
    if not is_success:
        out = np.zeros(dim_out)
    else:
        out = np.array([res.total,np.mean(res.iota), surf.aspect_ratio()])

    return out

def evaluate_configuration_winding_surface(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, M=10, N=10, G=1, ntheta=31, nphi=31, extend_factor=5):
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
        extend_factor (float, optional): factor by which to extend the winding surface in the normal direction. Defaults to 5.

    Returns:
        tuple: (quasisymmetry error, iota)
    """

    # build the surface
    stellsym_factor = 2 if stellsym else 1
    ntheta = max(2*mpol+1, ntheta)
    nphi = max(2*ntor+1, nphi)
    quadpoints_phi = np.linspace(0, 1 / (stellsym_factor * nfp), nphi, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=False)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf.x = x

    assert 2*M+1 <= ntheta and N <= 2*nphi + 1, "M and N must be less than or equal to ntheta and nphi respectively."


    # build winding surface
    orientation = compute_orientation(surf)
    dist_extend = surf.minor_radius()*(orientation * extend_factor)
    surf_winding = surf.to_RZFourier().copy(range='field period')
    surf_winding.extend_via_normal(dist_extend)
    current = SheetCurrent(surf_winding, G, M, N)

    # solve
    current.fit(surf)

    # compute iota using the least-squares estimate
    iota = compute_iota(surf, current)

    # compute QS-error
    quasi = "QA" if helicity == 0 else "QH"
    qs_error = NonQuasiSymmetricRatio(surf, current, quasi=quasi).J()
    qs_error = np.sqrt(qs_error)

    # compute boozer residual
    _, residual_mse = boozer_residual(surf, current, iota)
    
    out = np.array([qs_error, iota, surf.aspect_ratio(), residual_mse])

    return out



if __name__ == "__main__":
    x = np.random.randn(661) # 661 corresponds to mpol=ntor=10
    evaluate_configuration(x, nfp=3, mpol=10, ntor=10, helicity_n=0, vmec_input=None)