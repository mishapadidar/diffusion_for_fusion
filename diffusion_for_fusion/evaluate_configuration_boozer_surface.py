import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier
from diffusion_for_fusion.boozer_surface_sheet_solve import BoozerSurfaceSheetSolve


def evaluate_configuration(x, iota, nfp, stellsym=True, mpol=10, ntor=10, helicity=0, G=1):
    """Evaluate a Boozer Surface configuration with Boozer Surface Sheet Solve and compute metrics.

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

    solver = BoozerSurfaceSheetSolve(surf, G, iota)

    # compute QS-error
    quasi = "QA" if helicity == 0 else "QH"
    qs_error = NonQuasiSymmetricRatio(surf, solver, quasi=quasi).J()
    qs_error = np.sqrt(qs_error)

    # compute boozer residual
    residual_mse = solver.boozer_residual_mse() # (nphi, ntheta, 3)

    metrics = {'qs_error': qs_error, 'aspect_ratio': surf.aspect_ratio(), 'boozer_residual_mse': residual_mse}

    return metrics


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
        B (array): array of shape (2*sDIM+1, 2*sDIM+1, 3) that contains the Boozer magnetic field on the surface.
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.  
        quasi_poloidal: string that determines the type of quasisymmetry to compute. 'QA' for quasi-axisymmetry,
            'QP' for quasi-poloidal symmetry, and 'QH' for quasi-helical symmetry.
    """

    # def __init__(self, boozer_surface, bs, sDIM=20, quasi='QA'):
    def __init__(self, in_surface, boozer_solver, sDIM=20, quasi='QA'):
        assert isinstance(in_surface, SurfaceXYZTensorFourier)
        assert (quasi == 'QA') or (quasi == 'QH') or (quasi == 'QP')

        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM+1, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas, dofs=in_surface.dofs)

        boozer_solver.surface = surface

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
        self.boozer_solver = boozer_solver

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

        B = self.boozer_solver.B() # (nphi, ntheta, 3)
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
    xyz = surf.gamma()
    B = BoozerSurfaceSheetSolve(surf, G, iota).B().reshape((-1, 3)) # (n, 3)
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
    modB = modB.reshape(xyz.shape[:-1])
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], facecolors=plt.cm.plasma((modB -np.min(modB))/ (np.max(modB) -np.min(modB))), alpha=1.0, label='surf')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    test_evaluate_configuration()