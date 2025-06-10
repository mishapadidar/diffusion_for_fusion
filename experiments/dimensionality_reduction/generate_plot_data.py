import numpy as np
import pandas as pd
from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from sklearn.decomposition import PCA
import os

n_samples = 200

# load the data
Y = pd.read_pickle('../../data/QUASR.pkl') # y-values
X = np.load('../../data/dofs.npy') # x-values
Y = Y.reset_index(drop=True)

# vacuum solver params
M = N = 10
ntheta = nphi = 31
default_extend_factor = 0.20

# only evaluate a subset of the data
idx_samples = np.random.choice(len(X), n_samples, replace=False)
Y = Y.iloc[idx_samples].reset_index(drop=True)

# n_pca_components = [np.shape(X)[1], 330, 150, 75, 36, 18, 9, 4, 2]
n_pca_components = [np.shape(X)[1], 400, 200, 100, 50, 25, 12, 6, 3]

for jj, n_pca in enumerate(n_pca_components):

    # do dimensionality reduction
    if n_pca == np.shape(X)[1]:
        X_pca = np.copy(X)
    else:
        pca = PCA(n_components=n_pca, svd_solver='full')
        X_pca = pca.fit_transform(X)
        X_pca = pca.inverse_transform(X_pca)

    X_pca = X_pca[idx_samples]


    # storage
    data = {
        'qs_error': np.zeros(n_samples),
        'iota': np.zeros(n_samples),
        'aspect_ratio': np.zeros(n_samples),
        'boozer_residual_mse': np.zeros(n_samples),
        'M': np.zeros(n_samples),
        'N': np.zeros(n_samples),
        'ntheta': np.zeros(n_samples),
        'nphi': np.zeros(n_samples),
        'extend_factor': np.zeros(n_samples),
        'G': np.zeros(n_samples),
        'helicity': np.zeros(n_samples),
        'nfp': np.zeros(n_samples),
        'ID': Y.ID.values,
        'n_pca': n_pca*np.ones(n_samples, dtype=int)
    }

    for ii, xx in enumerate(X_pca):
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

        # collect the data
        data['qs_error'][ii] = metrics['qs_error']
        data['iota'][ii] = metrics['iota']
        data['aspect_ratio'][ii] = metrics['aspect_ratio']
        data['boozer_residual_mse'][ii] = metrics['boozer_residual_mse']
        data['M'][ii] = M
        data['N'][ii] = N
        data['ntheta'][ii] = ntheta
        data['nphi'][ii] = nphi
        data['extend_factor'][ii] = extend_factor
        data['G'][ii] = G
        data['helicity'][ii] = helicity
        data['nfp'][ii] = nfp
        print(f"Device {data['ID'][ii]}: qs_error={metrics['qs_error']}, iota={metrics['iota']}, aspect_ratio={metrics['aspect_ratio']}")

    # save data
    outdir = "./output/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfilename = outdir + f'evaluations_pca_{n_pca}.pkl'
    # append to existing file if it exists
    if os.path.exists(outfilename):
        existing_df = pd.read_pickle(outfilename)
        df = pd.concat([existing_df, pd.DataFrame(data)], ignore_index=True)
    else:
        df = pd.DataFrame(data)
    pd.to_pickle(df, outfilename)
    print(df.head())
    print(f"Data saved to {outfilename}")