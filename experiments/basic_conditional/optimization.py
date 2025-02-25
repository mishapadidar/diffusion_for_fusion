import numpy as np
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve

mpi = MpiPartition()

vmec = Vmec("../../vmec_input_files/input.nfp4_torus", mpi=mpi, verbose=False, keep_all_files=False)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 1
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)") # Major radius

# surf.plot()
# quit()

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.residuals, 0, 1)])

print("Quasisymmetry objective before optimization:", qs.total())
print("Total objective before optimization:", prob.objective())

least_squares_mpi_solve(prob, mpi, grad=True, ftol=0.1)


print("Quasisymmetry objective after optimization:", qs.total())
print("Total objective after optimization:", prob.objective())