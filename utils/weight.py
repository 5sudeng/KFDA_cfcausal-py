import numpy as np

def get_weight(x, treatment, mode="ATE", e_x=None, density_ratio=None):
    """
    Calculate weights for treatment effect estimation.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The feature matrix.
    treatment : array-like, shape (n_samples,)
        The treatment assignment (binary: 0 or 1).

    mode : str, optional
        The mode of weight calculation. Options are "ATE", "ATT", "ATC", or "General".
    e_x : array-like, shape (n_samples,), optional
        Propensity scores for the treatment assignment.
        Required for all modes except "ATE".
    density_ratio : array-like, shape (n_samples,), optional
        Required if mode is "General".

    Returns
    -------
    weights : array, shape (n_samples,)
        The calculated weights.
    """
    assert treatment.shape[0] == x.shape[0], "Mismatch: 'treatment' and 'x' must have the same number of rows."

    if e_x is None:
        raise ValueError("e_x must be provided.")

    elif mode == "ATE":
        return treatment / e_x + (1 - treatment) / (1 - e_x)

    elif mode == "ATT":
        return np.ones_like(treatment) * treatment + (1 - treatment) * (e_x / (1 - e_x))

    elif mode == "ATC":
        return np.ones_like(treatment) * (1 - treatment) + treatment * ((1 - e_x) / e_x)

    elif mode == "General":
        if density_ratio is None:
            raise ValueError("density_ratio must be provided for General mode.")
        return treatment * (density_ratio / e_x) + (1 - treatment) * (density_ratio / (1 - e_x))

    else:
        raise ValueError("Invalid mode. Choose from 'ATE', 'ATT', 'ATC', 'General'.")