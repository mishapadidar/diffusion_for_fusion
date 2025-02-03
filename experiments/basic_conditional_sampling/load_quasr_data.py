import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def load_quasr_data(conditions, return_pca=False):
    """
    Loads and processes QUASR dataset, optionally performing PCA transformation.
    
    Parameters:
        conditions (list): A subset of stellarator properties, list of conditions to use.
            i.e. ['mean_iota', 'aspect_ratio']. Select from,
            ['qs_error', 'coil_length_per_hp', 'total_coil_length',
             'total_coil_length_threshold', 'mean_iota', 'max_kappa', 'max_msc',
             'min_coil2coil_dist', 'nc_per_hp', 'nfp', 'constraint_success',
             'aspect_ratio', 'ID', 'minor_radius', 'Nfourier_coil', 'Nsurfaces',
             'volume', 'min_coil2surface_dist', 'mean_elongation', 'max_elongation',
             'message', 'iota_profile', 'volume_profile', 'tf_profile',
             'surface_types', 'currents', 'min_coil2axis_dist', 'axis_Rc', 'axis_Zs',
             'helicity']
        return_pca (bool): If True, returns PCA-transformed data instead of raw
    
    Returns:
        array-like: Raw data (X) or PCA-transformed data (X_pca)
        PCA object: Only if return_pca_components=True
    """

    # to load the data set
    Y_init = pd.read_pickle('../../data/QUASR.pkl') # y-values
    X_init = np.load('../../data/dofs.npy') # x-values

    Y_init = Y_init.reset_index(drop=True)

    # subset data
    idx = ((Y_init.nfp == 4) & (Y_init.helicity == 1))
    
    Y = Y_init[idx][conditions].values
    X = X_init[idx]

    # take PCA
    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(X)

    if return_pca:
        return X_pca, Y, pca
    else:
        return X, Y, pca
    
def plot_pca_data(X, X_new=None, is_pca=False, save_path=""):
    """
    Plots 2D PCA visualization of input data, optionally comparing two datasets.
    
    Parameters:
        X (array-like): Primary dataset to visualize
        X_new (array-like, optional): Secondary dataset to compare against X
        is_pca (bool): If True, assumes X is already PCA-transformed
        save_path (str): If provided, saves plot to this filepath
    
    Returns:
        None. Displays or saves matplotlib plot.
    """

    if not is_pca:
        # take PCA
        pca = PCA(n_components=2, svd_solver='full')
        X_pca = pca.fit_transform(X)
        if X_new is not None:
            X_new_pca = pca.transform(X_new)
    else:
        X_pca = X
        X_new_pca = X_new

    # plot the PCA
    fig,ax = plt.subplots()
    plt.scatter(X_pca[:,0], X_pca[:,1], color='tab:blue')
    if X_new is not None:
        plt.scatter(X_new_pca[:, 0], X_new_pca[:, 1], color='tab:orange')
    ax.set_aspect('equal')

    # plt.show()

    if save_path != "":
        fig.savefig(save_path) 
    

if __name__ == "__main__":
    X,Y, pca = load_quasr_data(["mean_iota", "aspect_ratio"], return_pca=True)
    X_new = X[:10]
    plot_pca_data(X, X_new = X_new, is_pca=True)
    plt.show()
