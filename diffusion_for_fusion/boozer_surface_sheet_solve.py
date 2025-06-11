import numpy as np

class BoozerSurfaceSheetSolve:
    """
    Find the (vacuum) magnetic field on a Boozer surface.

    A surface x(phi, theta) with Boozer angles satisfies
        G * B = |B|^2 * (dx/dphi + iota * dx/dtheta). (1)
    Taking the norm of both sides,
        |B| = G / |dx/dphi + iota * dx/dtheta|. (2)
    Equation (1) shows the direction of the magnetic field and equation (2) gives its magnitude.
    Combinging the two, the magnetic field is given by
        B = G * (dx/dphi + iota * dx/dtheta) / |dx/dphi + iota * dx/dtheta|^2.    (3)

    This class solves for the magnetic field on a Boozer surface given the surface using equation (3).

    Args:
        surface (SurfaceXYZTensorFourier): A surface with Boozer angles.
        G (float): Normalized poloidal current.
        iota (float): Rotation transform on the surface
    
    Returns:
        B (array): The magnetic field on the surface, shape (nphi, ntheta, 3).
    """
    def __init__(self, surface, G, iota):
        self.surface = surface
        self.G = G
        self.iota = iota
        self.need_to_run_code = True

    def B(self):
        dx_by_dphi = self.surface.gammadash1() # (nphi, ntheta, 3)
        dx_by_dtheta = self.surface.gammadash2()
        tangent = dx_by_dphi + self.iota * dx_by_dtheta # (nphi, ntheta, 3)
        norm_tangent = np.sum(tangent**2, axis=-1, keepdims=True)  # (nphi, ntheta, 1)
        B = self.G * tangent / norm_tangent**2
        return B
    
    def boozer_residual(self):
        """
        Compute the residual of the Boozer equation on the surface,
            r = G * B - |B|^2 * (dx/dphi + iota * dx/dtheta).

        Returns:
            residual (array): The residual of the Boozer equation, shape (nphi, ntheta, 3).
        """
        dx_by_dphi = self.surface.gammadash1() # (nphi, ntheta, 3)
        dx_by_dtheta = self.surface.gammadash2()
        tangent = dx_by_dphi + self.iota * dx_by_dtheta # (nphi, ntheta, 3)
        B = self.B()
        modB_squared = np.sum(B**2, axis=-1, keepdims=True) 
        residual = self.G * B - modB_squared * tangent
        return residual
    
    def boozer_residual_mse(self):
        """ Compute the mean squared error of the Boozer residual over the surface.

        Returns:
            total (float): The mean squared error of the Boozer residual over the surface.
        """
        dtheta = np.diff(self.surface.quadpoints_theta)[0]
        dphi = np.diff(self.surface.quadpoints_phi)[0]
        normal = self.surface.normal().reshape(-1, 3) # (n, 3)
        dA = np.linalg.norm(normal, axis=-1) * (dtheta * dphi) # (n)

        residual = self.boozer_residual().reshape((-1, 3)) # (n, 3)
        residual = np.linalg.norm(residual, axis=-1)  # (n,)
        total = np.sqrt(np.mean(np.sum(residual**2, axis=-1) * dA)) # (scalar)
        return total

    
def test_boozer_surface_sheet_solve():
    """Test the boozer_surface_sheet_solve function by loading a data point from QUASR."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from simsopt.geo import SurfaceXYZTensorFourier
    from diffusion_for_fusion.evaluate_configuration import evaluate_configuration

    # parameters
    M = N = 10
    extend_factor = 0.1
    nphi = ntheta = 31
    # which data point
    idx_data = 0

    # load the data set
    Y_init = pd.read_pickle('../data/QUASR.pkl') # y-values
    X_init = np.load('../data/dofs.npy') # x-values
    Y_init = Y_init.reset_index(drop=True)
    print(Y_init.columns)
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
    iota = Y.iota_profile[idx_data][-1] # rotational transform

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

    # compare to winding surface solve
    xyz = surf.gamma()
    B_ws = current.B(xyz.reshape((-1, 3))).reshape(xyz.shape) # (nphi, ntheta, 3)
    B_boozer = BoozerSurfaceSheetSolve(surf, G, iota).B() # (nphi, ntheta, 3)
    err = B_ws - B_boozer
    err_from_B_ws = np.max(np.abs(err))
    print('error from winding surface field', err_from_B_ws)

    # check the boozer residual is zero
    dtheta = np.diff(surf.quadpoints_theta)[0]
    dphi = np.diff(surf.quadpoints_phi)[0]
    normal = surf.normal().reshape(-1, 3) # (n, 3)
    dA = np.linalg.norm(normal, axis=-1) * (dtheta * dphi) # (n)
    dSigma_dphi = surf.gammadash1().reshape(-1, 3) # (n, 3)
    dSigma_dtheta = surf.gammadash2().reshape(-1, 3) # (n, 3)
    B = B_boozer.reshape((-1, 3)) # (n, 3)
    modB = np.linalg.norm(B, axis=-1) # (n)
    residual = G * B - (modB**2)[:, None] * (dSigma_dphi + iota * dSigma_dtheta)
    residual = np.linalg.norm(residual, axis=-1)  # (n,)
    total = np.sqrt(np.mean(np.sum(residual**2,axis=-1) * dA)) # (scalar)
    worst_residual = np.max(np.abs(residual))
    residual_mse = total
    print('residual_mse', residual_mse)
    print('worst_residual', worst_residual)

if __name__ == "__main__":
    test_boozer_surface_sheet_solve()