from sklearn.neighbors import KernelDensity
import numpy as np

def compute_kde(glitch_type, snr_series_of_each_class):
    
    """
    Compute a KDE for a given set of SNRs for a glitch.
    
    Parameters:
    -----------
    glitch_type: string
        identified for GravitySpy glitch class
        
    Returns:
    --------
    kde_skl: scikit-learn KernelDensity object
        fit KDE for a given glitch class
    """


    # Make KDE of one gravity spy distribution
    kde_skl = KernelDensity(bandwidth=1)
    kde_skl.fit(np.asarray(snr_series_of_each_class[glitch_type]).reshape(-1, 1))

    return kde_skl
