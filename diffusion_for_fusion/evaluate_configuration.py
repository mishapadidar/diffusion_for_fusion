import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual

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

    qs = QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1,1,10,endpoint=False), helicity_n=helicity_n)

    is_success = True
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

if __name__ == "__main__":
    x = np.random.randn(661) # 661 corresponds to mpol=ntor=10
    evaluate_configuration(x, nfp=3, mpol=10, ntor=10, helicity_n=0, vmec_input=None)