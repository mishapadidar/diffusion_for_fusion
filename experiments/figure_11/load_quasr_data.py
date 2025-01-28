import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def load_quasr_data(return_pca=False, fig="fig11"):
    """
    Loads and processes QUASR dataset, optionally performing PCA transformation.
    
    Parameters:
        return_pca (bool): If True, returns PCA-transformed data instead of raw
        fig (str): Dataset subset to use - either "fig9" or "fig11"
    
    Returns:
        array-like: Raw data (X) or PCA-transformed data (X_pca)
        PCA object: Only if return_pca_components=True
    """

    # to load the data set
    Y_init = pd.read_pickle('QUASR.pkl') # y-values
    X_init = np.load('dofs.npy') # x-values

    Y_init = Y_init.reset_index(drop=True)

    if fig == "fig11":
        # subset figure 11 dat
        idx = ((np.abs(Y_init.mean_iota - 2.30)/2.30 < 0.001) & (np.abs(Y_init.aspect_ratio - 12)/12 < 0.1)
            & (Y_init.nfp == 4) & (Y_init.helicity == 1))
    elif fig == "fig9":
        # subset figure 9 dat
        idx = ((np.abs(Y_init.mean_iota - 1.1)/1.1 < 0.001) & (np.abs(Y_init.aspect_ratio - 12)/12 < 0.01)
            & (Y_init.nfp == 4) & (Y_init.helicity == 1))
    else:
        raise ValueError("fig can be one of fig9 or fig11")
    
    Y = Y_init[idx].mean_iota.values
    X = X_init[idx]

    # take PCA
    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(X)

    if return_pca:
        return X_pca, pca
    else:
        return X, pca
    
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
    X = load_quasr_data(return_pca=True,fig_num=9)
    X_new = X[:10]
    plot_pca_data(X, X_new = X_new, is_pca=True)
