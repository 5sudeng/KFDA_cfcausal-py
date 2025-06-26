import numpy as np

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False):
    """
    Compute the weighted quantiles of a set of values.

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

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    cumulative_weights = np.cumsum(sample_weight)
    total_weight = cumulative_weights[-1]

    quantile_values = []
    for q in quantiles:
        target_weight = q * total_weight
        index = min(np.searchsorted(cumulative_weights, target_weight), len(values) - 1)
        quantile_values.append(values[index])

    return quantile_values[0] if len(quantile_values) == 1 else np.array(quantile_values)