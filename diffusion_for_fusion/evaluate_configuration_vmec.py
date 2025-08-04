import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.mhd import Vmec, Quasisymmetry, Boozer, QuasisymmetryRatioResidual

def evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, nphi=51, ntheta=51, vmec_input="input.nfp4_template"):
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
    # surf = SurfaceXYZTensorFourier(
    #     mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
    #     quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False),
    #     quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    surf = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp)
    surf.x = x

    # vmec = Vmec(vmec_input, verbose=False, keep_all_files=False, range_surface='field period')
    vmec = Vmec(vmec_input, verbose=False, keep_all_files=False)
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
        iota_edge = vmec.iota_edge()
        mean_iota = vmec.mean_iota()

        # compute QS-error
        s = 1.0
        try:
            boozer = Boozer(vmec)
            qs = Quasisymmetry(boozer,
                            np.linspace(0.1,1,10,endpoint=False), # Radius to target
                            helicity_m=1,
                            helicity_n=helicity,
                            normalization='symmetric',
                            weight='stellopt_ornl')
            sqrt_qs_error_boozer = np.mean(qs.J())
        except:
            sqrt_qs_error_boozer = np.nan

        qs_2term = QuasisymmetryRatioResidual(vmec, surfaces=np.linspace(0.1,1,10,endpoint=False), helicity_m=1, helicity_n= helicity)
        res = qs_2term.compute()
        sqrt_qs_error_2term = np.sqrt(res.total)
        # mean_iota = np.mean(res.iota)

        try:
            sqrt_non_qs_error = BoozerNonQuasiSymmetricRatio(boozer, s=s, helicity=0, nphi=nphi, ntheta=ntheta)
        except:
            sqrt_non_qs_error = np.nan
    else:
        # fallback if VMEC fails
        iota_edge = np.nan
        mean_iota = np.nan
        sqrt_qs_error_boozer = np.nan
        sqrt_qs_error_2term = np.nan
        sqrt_non_qs_error = np.nan

    metrics = {'sqrt_qs_error_boozer': sqrt_qs_error_boozer,
               'sqrt_qs_error_2term': sqrt_qs_error_2term, 
               'sqrt_non_qs_error': sqrt_non_qs_error,
               'iota_edge': iota_edge,
               'mean_iota': mean_iota, 
               'aspect_ratio': aspect_ratio,
               'success': success}

    return metrics, vmec


def BoozerNonQuasiSymmetricRatio(boozer, s=1.0, helicity=0, nphi=31, ntheta=31):
    """Compute the NonQuasiSymmetricRatio from a Boozer object.

    Args:
        boozer (Boozer): Boozer object.
        s (float): surface to evaluate the NonQuasiSymmetricRatio on.
        helicity (int, optional): 0 or 1 to indicate QA or QH. Defaults to 0.
        nphi (int, optional): number of phi quadrature points. Defaults to 31.
        ntheta (int, optional): number of theta quadrature points. Defaults to 31.

    Returns:
        float: square root of non-quasisymmetric ratio metric
    """
    boozer.register([s])
    boozer.run()

    # get boozXform object
    bx = boozer.bx

    # discretize boozer angles
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d = np.linspace(0, 2 * np.pi / bx.nfp, nphi)
    phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')

    # index of flux surface
    js = 0

    # reconstruct |B| and sqrtg
    modB = np.zeros(np.shape(phi))
    sqrtg = np.zeros(np.shape(phi))
    for jmn in range(len(bx.xm_b)):
        m = bx.xm_b[jmn]
        n = bx.xn_b[jmn]
        angle = m * theta - n * phi
        modB += bx.bmnc_b[jmn, js] * np.cos(angle)
        sqrtg += bx.gmnc_b[jmn, js] * np.cos(angle)
        if bx.asym:
            modB += bx.bmns_b[jmn, js] * np.sin(angle)
            sqrtg += bx.gmns_b[jmn, js] * np.sin(angle)

    def make_QA_matrix(in_nphi, in_ntheta):
        idx = np.arange(in_nphi)
        jdx = np.arange(in_ntheta)
        idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
        return idx, jdx

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
    
    if helicity == 1:
        idx, jdx = make_QH_matrix(nphi, ntheta)
    else:
        idx, jdx = make_QA_matrix(nphi, ntheta)

    # Andrew's non-QS ratio
    dS = sqrtg[idx, jdx]
    modB = modB[idx, jdx]
    B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
    B_QS = B_QS[None, :]
    B_nonQS = modB - B_QS
    _J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)
    _J = np.sqrt(_J)
    return _J



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
    # idx_data = 1203
    idx_data = 368248
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
    metrics, _ = evaluate_configuration(x, nfp, stellsym=True, mpol=10, ntor=10, helicity=Y.helicity[idx_data], ntheta=31, nphi=31)
    print('metrics', metrics)


if __name__ == "__main__":
    test_evaluate_configuration()