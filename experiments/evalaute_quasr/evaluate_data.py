import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration

# load the data
# to load the data set
Y = pd.read_pickle('../../data/QUASR.pkl') # y-values
X = np.load('../../data/dofs.npy') # x-values
Y = Y.reset_index(drop=True)

# TODO: use adaptive M,N,ntheta,nphi depending on nfp!
# vacuum solver params
M = N = 10
ntheta = nphi = 31
default_extend_factor = 0.20

for ii, xx in enumerate(X):
    nfp = Y.nfp[ii]
    I_P = sum(np.abs(Y.currents[0])) * nfp * 2 # total current through hole
    G = (4 * np.pi * 1e-7) * I_P
    helicity = Y.helicity[ii]
    
    extend_factor = default_extend_factor
    metrics, _ = evaluate_configuration(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=M, N=N, G=G, ntheta=ntheta, nphi=nphi, extend_factor=extend_factor)

    # reevaluate if results are bad: probably due to self-intersecting winding surface
    if np.abs(metrics['iota'] - Y.iota_profile[ii][-1]) > 0.05 :
        extend_factor = default_extend_factor / 2
        metrics, _ = evaluate_configuration(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=M, N=N, G=G, ntheta=ntheta, nphi=nphi, extend_factor=extend_factor)

    qs_err_actual = np.sqrt(Y.qs_error[ii])
    iota_actual = Y.iota_profile[ii][-1]
    aspect_ratio_actual = Y.aspect_ratio[ii]
    print("")
    print(f"Configuration {ii}) nfp={nfp}, helicity={helicity}:")
    print(f"Actual values: qs_err={qs_err_actual}, iota={iota_actual}, aspect_ratio={aspect_ratio_actual}")
    print(f"Evaluated values: qs_err={metrics['qs_error']}, iota={metrics['iota']}, aspect_ratio={metrics['aspect_ratio']}")

    # modify Y
    Y.loc[ii, 'qs_error_vacuum_solver'] = metrics['qs_error']
    Y.loc[ii, 'iota_vacuum_solver'] = metrics['iota']
    Y.loc[ii, 'aspect_ratio_vacuum_solver'] = metrics['aspect_ratio']
    Y.loc[ii, 'boozer_residual_mse_vacuum_solver'] = metrics['boozer_residual_mse']
    Y.loc[ii, 'M_vacuum_solver'] = M
    Y.loc[ii, 'N_vacuum_solver'] = N
    Y.loc[ii, 'ntheta_vacuum_solver'] = ntheta
    Y.loc[ii, 'nphi_vacuum_solver'] = nphi
    Y.loc[ii, 'extend_factor_vacuum_solver'] = extend_factor

outfilename = '../../data/QUASR_reevaluated.pkl'
pd.to_pickle(Y, outfilename)
print(Y.head())
print(f"Re-evaluated QUASR data saved to {outfilename}")