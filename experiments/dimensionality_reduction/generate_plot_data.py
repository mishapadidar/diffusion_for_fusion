import numpy as np
import pandas as pd
# from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
from diffusion_for_fusion.evaluate_configuration_boozer_surface import evaluate_configuration as evaluate_configuration_boozer
from diffusion_for_fusion.evaluate_configuration_sheet_curent import evaluate_configuration as evaluate_configuration_sheet
from sklearn.decomposition import PCA
import os
import time

n_samples = 50


# load the data
Y = pd.read_pickle('../../data/QUASR.pkl') # y-values
X = np.load('../../data/dofs.npy') # x-values
Y = Y.reset_index(drop=True) # this is critical for indexing.

# only evaluate a subset of the data
idx_samples = np.random.choice(len(X), n_samples, replace=False)
Y = Y.iloc[idx_samples].reset_index(drop=True)

n_full = np.shape(X)[1]
n_pca_components = [n_full, 400, 300, 200, 150, 100, 50, 40, 30, 25, 17, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

for jj, n_pca in enumerate(n_pca_components):
    print(f"Evaluating PCA with {n_pca} components...")

    # do dimensionality reduction
    if n_pca == n_full:
        X_pca = np.copy(X)
    else:
        pca = PCA(n_components=n_pca, svd_solver='full')
        X_pca = pca.fit_transform(X)
        X_pca = pca.inverse_transform(X_pca)

    X_pca = X_pca[idx_samples]


    # storage
    data = {
        'sqrt_qs_error': np.zeros(n_samples),
        'iota': np.zeros(n_samples),
        'aspect_ratio': np.zeros(n_samples),
        'boozer_residual_mse': np.zeros(n_samples),
        'G': np.zeros(n_samples),
        'helicity': np.zeros(n_samples),
        'nfp': np.zeros(n_samples),
        'ID': Y.ID.values,
        'n_pca': n_pca*np.ones(n_samples, dtype=int)
    }

    t0 = time.time()
    for ii, xx in enumerate(X_pca):
        if ii % 10 == 0:
            t1 = time.time()
            print(f"Evaluating configuration {ii}/{n_samples}; time elapsed: {(t1 - t0)/(ii+1):.2f} s per configuration")
        nfp = Y.nfp[ii]
        I_P = sum(np.abs(Y.currents[ii])) * nfp * 2 # total current through hole
        G = (4 * np.pi * 1e-7) * I_P
        helicity = Y.helicity[ii]
        iota = Y.iota_profile[ii][-1] # rotational transform
        
        # metrics = evaluate_configuration_boozer(xx, iota, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, G=G)
        metrics, _ = evaluate_configuration_sheet(xx, nfp, stellsym=True, mpol=10, ntor=10, helicity=helicity, M=10, N=10, G=G, ntheta=31, nphi=31, extend_factor=0.1)

        qs_err_actual = np.sqrt(Y.qs_error[ii])
        aspect_ratio_actual = Y.aspect_ratio[ii]

        # print("")
        # print(f"Configuration {ii}) Device {data['ID'][ii]}, nfp={nfp}, helicity={helicity}:")
        # print(f"QUASR metrics: qs_err={qs_err_actual}, iota={iota}, aspect_ratio={aspect_ratio_actual}")
        # print(f"Evaluated metrics: qs_err={metrics['sqrt_qs_error']}, aspect_ratio={metrics['aspect_ratio']}, boozer_residual_mse={metrics['boozer_residual_mse']}")

        # collect the data
        data['sqrt_qs_error'][ii] = metrics['sqrt_qs_error']
        data['iota'][ii] =  metrics['iota']
        data['aspect_ratio'][ii] = metrics['aspect_ratio']
        data['boozer_residual_mse'][ii] = metrics['boozer_residual_mse']
        data['G'][ii] = G
        data['helicity'][ii] = helicity
        data['nfp'][ii] = nfp

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

