import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.mhd import Vmec, Quasisymmetry, Boozer

def evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, nphi=51, ntheta=51):
    """Evaluate a surface configuration with VMEC.

    QUASR data uses mpol=ntor=10 and stellsym=True.

    Args:
        x (np array)): 1d array of degrees of freedom for a SurfaceXYZTensorFourier.
        nfp (int): number of field periods.
        stellsym (bool, optional): True for stellarator symmetric configurations. Defaults to True.
        mpol (int): number of poloidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        ntor (int): number of toroidal modes for the surface (not winding surface). The number of degrees of freedom in x must match with mpol and ntor.
        helicity (int, optional): quasisymmetry helicity n. Defaults to 0 for QA. Use 1 or -1 for QH.
        ntheta (int, optional): minimum number of surface (not winding surface) quadrature points in theta direction. 
            The actual ntheta will be max(2*mpol +1, ntheta). Defaults to 31.
        nphi (int, optional): minimum number of surface (not winding surface) quadrature points in phi direction.
            The actual nphi will be max(2*mpol +1, nphi). Defaults to 31.

    Returns:
        tuple: (metrics, sheet_current)
        metrics (dict): dictionary of metrics
        sheet_current (SheetCurrent): SheetCurrent object with the fitted current.
    """

    # build the surface
    nphi = max(2*ntor+1, nphi)
    ntheta = max(2*mpol+1, ntheta)
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    surf.x = x

    vmec_input = "input.nfp4_template"
    vmec = Vmec(vmec_input, verbose=True, keep_all_files=False, range_surface='field period')
    # change nfp
    vmec.indata.nfp = nfp
    # run
    vmec.boundary = surf

    # # plot it
    # surfrz = surf.to_RZFourier()
    # xyz = surfrz.gamma()
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2])
    # plt.show()

    success=True
    try:
        vmec.run()
    except:
        success=False
    
    aspect_ratio = surf.aspect_ratio()
    if success:
        # compute iota
        iota = vmec.iota_edge()

        # compute QS-error
        boozer = Boozer(vmec)
        qs = Quasisymmetry(boozer,
                        1.0, # Radius to target
                        helicity_m=1,
                        helicity_n=helicity,
                        normalization='symmetric',
                        weight='stellopt_ornl') # (M, N) you want in |B|
        qs_error = qs.J().item()
    else:
        # fallback if VMEC fails
        iota = np.nan
        qs_error = np.nan

    metrics = {'sqrt_qs_error': qs_error, 'iota': iota, 'aspect_ratio': aspect_ratio, 'success': success}

    return metrics


def test_evaluate_configuration():
    """Test the evaluate_configuration function by loading a data point from QUASR."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # load the data set
    Y_init = pd.read_pickle('../data/QUASR.pkl') # y-values
    X_init = np.load('../data/dofs.npy') # x-values
    Y_init = Y_init.reset_index(drop=True)
    conditions = ["qs_error", 'iota_profile', "aspect_ratio", "nfp", "helicity", "currents", 'nc_per_hp']
    Y = Y_init[conditions]
    X = X_init

    # which data point
    # idx_data = 7
    # idx_data = 35
    idx_data = 1203
    # idx_data = 368248
    # idx_data = 212450
    # idx_data = 110825
    # idx_data = 239324

    print("nfp", Y.nfp[idx_data])
    print("helicity", Y.helicity[idx_data])
    print("edge iota", Y.iota_profile[idx_data][-1])
    print("aspect ratio", Y.aspect_ratio[idx_data])
    print("sqrt qs error", np.sqrt(Y.qs_error[idx_data]))

    # pick a data point
    x = X[idx_data]
    nfp = Y.nfp[idx_data]

    # evaluate the configuration
    metrics = evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=Y.helicity[idx_data], ntheta=31, nphi=31)
    print('metrics', metrics)


if __name__ == "__main__":
    test_evaluate_configuration()