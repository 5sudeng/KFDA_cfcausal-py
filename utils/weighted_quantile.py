import numpy as np

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, w_x=None):
    """
    Compute the weighted quantiles of a set of values, optionally including a point mass at infinity.

    Parameters
    ----------
    values : array-like, shape (n_samples,)
        The input values for which to compute quantiles.
    quantiles : float or array-like, shape (n_quantiles,)
        The quantiles to compute, in the range [0, 1].
    sample_weight : array-like, shape (n_samples,), optional
        Weights for each value. If None, all values are equally weighted.
    values_sorted : bool, optional
        If True, 'values' is assumed to be sorted. Default is False.
    w_x : float, optional
        Weight of the test point. Default is None.

    Returns
    -------
    quantile_values : float or np.ndarray
        The computed quantile value(s).
    """
    values = np.asarray(values)
    quantiles = np.atleast_1d(quantiles)

    if sample_weight is None:
        sample_weight = np.ones_like(values)
    else:
        sample_weight = np.asarray(sample_weight)

    if values.shape[0] != sample_weight.shape[0]:
        raise ValueError(f"'values' and 'sample_weight' must have the same length. Got {values.shape[0]} and {sample_weight.shape[0]}")

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    quantile_values = []
    W = np.asarray(sample_weight)
    if w_x is None:
        w_x = 0.0

    if np.isscalar(w_x):
        Z = np.sum(W) + w_x
    elif isinstance(w_x, np.ndarray):
        # Allow shape () (scalar), shape (1,), or shape matching values
        if w_x.shape[0] != 1 and w_x.shape[0] != values.shape[0]:
            raise ValueError(f"'w_x' has shape {w_x.shape}, expected scalar or shape {values.shape}")
        Z = np.sum(W) + w_x  # Will broadcast if shape matches
    else:
        raise ValueError(f"Unsupported type for w_x: {type(w_x)}")

    p_i = W / Z
    p_inf = w_x / Z

    for q in quantiles:
        cumulative_weights = np.cumsum(p_i)
        if p_inf > 1 - q:
            quantile_values.append(np.inf)
            continue
        adjusted_q = q / (1 - p_inf)
        target_weight = adjusted_q
        index = min(np.searchsorted(cumulative_weights, target_weight), len(values) - 1)
        quantile_values.append(values[index])

    return quantile_values[0] if len(quantile_values) == 1 else np.array(quantile_values)