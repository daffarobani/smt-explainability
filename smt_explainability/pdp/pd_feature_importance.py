import numpy as np
from .partial_dependence import partial_dependence


def compute_pd_feature_importance(features, is_categorical, pd_results):
    importances = []
    for i, feature in enumerate(features):
        pd = pd_results[i]["average"]
        if is_categorical[feature]:
            importance = (np.max(pd) - np.min(pd)) / 4
        else:
            k = len(pd)
            mean_pd = np.mean(pd)
            importance = np.power(np.sum((pd - mean_pd) ** 2) / (k - 1), 0.5)
        importances.append(importance)
    return importances


def pd_feature_importance(
    model,
    x,
    features,
    *,
    categorical_feature_indices=None,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    method="uniform",
    ratio_samples=None,
):
    """
    Computes feature importance based on partial dependence.

    Args:
        model (object): The model used for predictions.
        x (numpy.ndarray): Data used for partial dependence computation.
        features (list): List of feature indices.
        categorical_feature_indices (list, optional): Indices of categorical features.
        percentiles (tuple, optional): Percentiles used to compute the grid for continuous features.
        grid_resolution (int, optional): Number of points in the grid for continuous features.
        method (str, optional): Method to use for grid computation ('sample', 'unique', or 'uniform').
        ratio_samples (float, optional): Ratio of samples to use for computing partial dependence.

    Returns:
        list: Computed feature importances.
    """
    pd_results = partial_dependence(
        model,
        x,
        features,
        categorical_feature_indices=categorical_feature_indices,
        percentiles=percentiles,
        grid_resolution=grid_resolution,
        method=method,
        kind="average",
        ratio_samples=ratio_samples,
    )

    is_categorical = [False] * x.shape[1]
    if categorical_feature_indices is not None:
        for feature_idx in categorical_feature_indices:
            is_categorical[feature_idx] = True

    importances = compute_pd_feature_importance(features, is_categorical, pd_results)
    return importances
