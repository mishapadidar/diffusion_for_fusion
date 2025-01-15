import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def figure_11_data(return_pca=False, standardize=True, plot=False, X_new=None, save_path=None,
                   return_pca_components=False):
    """
    Performs PCA on fusion reactor data filtered by specific constraints 
    (mean_iota ≈ 2.30, aspect_ratio ≈ 12, nfp = 4, helicity = 1).
    
    Parameters
    ----------
    return_pca : bool, optional
        If True, returns PCA-transformed data instead of original data
    standardize : bool, optional
        Flag for data standardization (currently inactive)
    plot : bool, optional
        If True, generates and saves PCA scatter plot
    X_new : array-like, optional
        New data points to transform with PCA
    save_path : str, optional
        Path to save plot if plotting enabled
        
    Returns
    -------
    numpy.ndarray
        PCA-transformed or original filtered data based on return_pca
    """
    # to load the data set
    Y_init = pd.read_pickle('QUASR.pkl') # y-values
    X_init = np.load('dofs.npy') # x-values

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

    if X_new is not None:
        X_new = X_new.numpy()
        if return_pca:
            X_new_pca = X_new
        else:
            X_new_pca = X_new @ np.vstack((dir1, dir2)).T

    # if standardize:
    #     # lb = np.min(X, axis=0)
    #     # ub = np.max(X, axis=0)
    #     # X = (X - lb)/(ub - lb)

    #     lb_pca = np.min(X_pca, axis=0)
    #     ub_pca = np.max(X_pca, axis=0)
    #     X_pca = (X_pca - lb_pca)/(ub_pca - lb_pca)

    #     if X_new is not None:
    #         # X_new = (X_new - lb)/(ub - lb)
    #         X_new_pca = (X_new_pca - lb_pca)/(ub_pca - lb_pca)

    if plot:
        # plot the PCA
        fig,ax = plt.subplots()
        plt.scatter(X_pca[:,0], X_pca[:,1], color='tab:blue')
        if X_new is not None:
            plt.scatter(X_new_pca[:, 0], X_new_pca[:, 1], color='tab:orange')
        ax.set_aspect('equal')
        #plt.show()
        fig.savefig(save_path) 

    if return_pca:
        if return_pca_components:
            return X_pca, dir1, dir2
        else:
            return X_pca
    else:
        return X
    

if __name__ == "__main__":
    figure_11_data(return_pca=False, plot=True)
