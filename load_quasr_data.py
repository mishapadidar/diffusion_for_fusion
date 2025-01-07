import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def figure_11_data(quasr_file='QUASR.pkl', dofs_file='dofs.npy', return_pca=False, plot=False):
    """
    Loads and filters stellarator configuration data based on physical constraints,
    then performs 2D PCA dimensionality reduction.

    Filters data for:
    - Mean iota ≈ 2.30 (0.1% tolerance)
    - Aspect ratio ≈ 12 (10% tolerance)
    - nfp = 4
    - helicity = 1

    Parameters
    ----------
   quasr_file : str, optional (default='QUASR.pkl')
       Path to pickle file containing stellarator parameters DataFrame
   dofs_file : str, optional (default='dofs.npy')
       Path to numpy file containing degrees of freedom data
    return_pca : bool, optional (default=False)
        If True, returns PCA-transformed data (2D)
        If False, returns original filtered data
    plot : bool, optional (default=False)
        If True, creates scatter plot of data in PCA space

    Returns
    -------
    numpy.ndarray
        PCA-transformed (n_samples, 2) array if return_pca=True
        Original filtered (n_samples, n_features) array if return_pca=False

    Notes
    -----
    Requires 'QUASR.pkl' and 'dofs.npy' in working directory
    """
    # to load the data set
    Y_init = pd.read_pickle(quasr_file) # y-values
    X_init = np.load(dofs_file) # x-values

    Y_init = Y_init.reset_index(drop=True)

    # subset figure 11 dat
    idx = ((np.abs(Y_init.mean_iota - 2.30)/2.30 < 0.001) & (np.abs(Y_init.aspect_ratio - 12)/12 < 0.1)
        & (Y_init.nfp == 4) & (Y_init.helicity == 1))
    Y = Y_init[idx].mean_iota.values
    X = X_init[idx]

    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(X)

    # PCA components
    mean = pca.mean_
    dir1 = pca.components_[0]
    dir2 = pca.components_[1]

    if plot:
        fig,ax = plt.subplots()
        plt.scatter(X_pca[:,0], X_pca[:,1])
        ax.set_aspect('equal')
        plt.show()

    if return_pca:
        return X_pca
    else:
        return X
    

if __name__ == "__main__":
    figure_11_data(return_pca=False, plot=True)
