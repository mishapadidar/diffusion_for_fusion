import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_fusion import to_standard

def prepare_data_from_config(config):
    """Load and prepare the data for use in the diffuion model training or evaluation.
    This function
        1. loads the QUASR data with the right conditions.
        2. optionally take the PCA of the X tensor
        3. standardizes the data

    Args:
        config (_type_): argparse config.

    Returns:
        X_train (array): stellarator degrees of freedom; (n, input_dim).
        X_mean (array): mean of the columns of X; (input_dim,).
        X_std (array): standard deviation of the columns of X; (input_dim,).
        Y_train (array): conditions (stellarator properies); (n, cond_input_dim).
        Y_mean (array): mean of the columns of Y; (cond_input_dim,).
        Y_std (array): standard deviation of the columns of Y; (cond_input_dim,).
        pca (pca object): can be used to project data onto the principal components.
    """
    # X: (n, input_dim) array, Y_train: (n, cond_input_dim) array
    X_train, Y_train = load_quasr_data(conditions=config.conditions)

    # use projected data
    pca = PCA(n_components=config.pca_size, svd_solver='full')
    if config.use_pca:
        X_train = pca.fit_transform(X_train)
        print("\nTaking PCA of data")
        print("Percent explained variance:", sum(pca.explained_variance_ratio_))
        print("")

    # standardize the data
    X_train, X_mean, X_std = to_standard(X_train)
    Y_train, Y_mean, Y_std = to_standard(Y_train)

    return X_train, X_mean, X_std, Y_train, Y_mean, Y_std, pca

def load_quasr_data(conditions):
    """
    Loads and processes QUASR dataset.
    
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
    
    Returns:
        (array): Raw data, X, of QUASR degrees-of-freedom; shape (n, input_dim).
        (array): Raw data, Y, of stellarator properties; shape (n, len(conditions)).
    """

    # to load the data set
    Y_init = pd.read_pickle('../../data/QUASR.pkl') # y-values
    X_init = np.load('../../data/dofs.npy') # x-values

    Y_init = Y_init.reset_index(drop=True)

    # subset data
    idx = ((Y_init.nfp == 4) & (Y_init.helicity == 1))
    
    Y = Y_init[idx][conditions].values
    X = X_init[idx]

    return X, Y
    
def plot_pca_data(X, X_new=None, is_pca=False, save_path=""):
    """
    Plots 2D PCA visualization of input data, optionally comparing two datasets.
    
    Parameters:
        X (array-like): Primary dataset to visualize.
        X_new (array-like, optional): Secondary dataset to compare against X.
        is_pca (bool): If True, assumes X is already PCA-transformed to two dimensions.
        save_path (str): If provided, saves plot to this filepath.
    
    Returns:
        fig, ax. Matplotlib figure and axis objects.
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

    return fig, ax
    

if __name__ == "__main__":
    X,Y, pca = load_quasr_data(["mean_iota", "aspect_ratio"], return_pca=True)
    X_new = X[:10]
    plot_pca_data(X, X_new = X_new, is_pca=True)
    plt.show()
