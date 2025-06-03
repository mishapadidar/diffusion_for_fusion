import numpy as np
from simsopt._core import Optimizable
from simsopt.geo import SurfaceXYZTensorFourier

def compute_nonQS(surface, sheet_current, helicity_n = 0):
    """Compute the NonQuasiSymmetricRatio metric from Simsopt for a given surface and sheet current.

    Args:
        surface (Surface): Surface object from Simsopt.
        sheet_current (SheetCurrent): SheetCurrent object
        helicity_n (int, optional): quasisymmetry helicity n. Defaults to 0 for QA. Should be 
            one of 0, 1, or -1.

    Returns:
        J (float): value of the NonQuasiSymmetricRatio metric.
    """
    if helicity_n == 0:
        axis = 0
    else:
        # TODO: this function is doing QA only right now.
        # TODO: we need to set up QH as well using the helical angle (theta - N*phi)
        raise NotImplementedError("NonQuasiSymmetricRatio for helicity_n != 0 is not implemented yet.")

    # compute J
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size

    B = sheet_current.B(surface.gamma().reshape((-1, 3))) # (n, 3)
    B = B.reshape((nphi, ntheta, 3))
    modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

    nor = surface.normal()
    dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

    B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

    if axis == 0:
        B_QS = B_QS[None, :]
    else:
        B_QS = B_QS[:, None]

    B_nonQS = modB - B_QS
    return np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)


def compute_iota(surf, current):
    """Compute the iota value from the surface and current by minimizing the boozer residual,

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
