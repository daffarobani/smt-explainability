"""Partial dependence for classification and regression models"""

# Authors: Muhammad Daffa Robani
import numpy as np
from scipy.stats.mstats import mquantiles
from typing import Union, List, Tuple


def grid_from_x(
    x,
    features,
    percentiles,
    grid_resolution,
    is_categorical,
    method,
):
    if isinstance(features, int):
        features = tuple([features])

    grid_values = []
    for i in features:
        if method == "sample":
            axis = x[:, i]
        elif method == "unique" or is_categorical[i]:
            axis = np.unique(x[:, i])
        elif method == "uniform":
            emp_percentiles = mquantiles(x[:, i], prob=percentiles, axis=0)
            axis = np.linspace(
                emp_percentiles[0],
                emp_percentiles[1],
                num=grid_resolution,
                endpoint=True,
            )
        else:
            raise ValueError("Method must be 'sample' / 'unique' / 'uniform'.")

        grid_values.append(axis)

    if method in ["unique", "uniform"]:
        grid_2d = cartesian(grid_values)
    else:
        grid_2d = non_cartesian(grid_values)
    return grid_2d, grid_values


def _partial_dependence_brute(
    model,
    grid_cartesian,
    grid_values,
    features,
    x,
    method,
    ratio_samples=None,
):
    if isinstance(features, int):
        features = tuple([features])

    nsamp = len(x)
    lengths = [len(grid_value) for grid_value in grid_values]
    predictions = []
    averaged_predictions = []
    if ratio_samples is None:
        nsamples = nsamp
        x_samp = x.copy()
    else:
        nsamples = int(ratio_samples * nsamp)
        index = np.random.choice(nsamp, nsamples, replace=False)
        x_samp = x[index]

    for new_values in grid_cartesian:
        x_eval = x_samp.copy()
        for i, feature in enumerate(features):
            x_eval[:, feature] = new_values[i]
        try:
            pred = model.predict_values(x_eval)
        except AttributeError:
            pred = model.predict(x_eval)
        averaged_pred = np.average(pred)

        predictions.append(pred)
        averaged_predictions.append(averaged_pred)

    predictions = np.array(predictions).T
    averaged_predictions = np.array(averaged_predictions).T

    if method != "sample":
        predictions = predictions.reshape([nsamples] + lengths)
        averaged_predictions = averaged_predictions.reshape(lengths)

    return averaged_predictions, predictions


def partial_dependence(
    model,
    x,
    features: Union[List, Tuple],
    *,
    categorical_feature_indices: List = None,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    kind="average",
    method="uniform",
    ratio_samples=None,
    categories_map=None,
):
    """
    Compute partial dependence.

    Args:
        model: The model used for predictions.
        x (numpy.ndarray): The data on which partial dependence is computed.
        features (Union[List, Tuple]): The features for which partial dependence is computed.
        categorical_feature_indices (List, optional): List of indices of categorical features.
        percentiles (tuple, optional): Percentiles used to compute the grid for continuous features.
        grid_resolution (int, optional): Number of points in the grid for continuous features.
        kind (str, optional): Type of partial dependence to compute ('average', 'individual', or 'both').
        method (str, optional): Method to use for grid computation ('sample', 'unique', or 'uniform').
        ratio_samples (float, optional): Ratio of samples to use for computing partial dependence.
        categories_map (dict, optional): Mapping of categorical feature values to their names.

    Returns:
        pdp_results (list): List of dictionaries containing partial dependence results for each feature.
    """
    for i, feature in enumerate(features):
        if type(feature) in [tuple, list]:
            if len(feature) == 1:
                features[i] = feature[0]
            elif len(feature) == 2:
                features[i] = tuple(feature)
            else:
                if method != "sample":
                    raise ValueError("Interaction features can't be more than two.")

    # list to store the features are categorical or not in x
    is_categorical = [False] * x.shape[1]
    if categorical_feature_indices is not None:
        for feature_idx in categorical_feature_indices:
            is_categorical[feature_idx] = True

    pdp_results = []
    for feature in features:
        # create grid
        grid_cartesian, grid_values = grid_from_x(
            x,
            feature,
            percentiles,
            grid_resolution,
            is_categorical,
            method,
        )

        # predictions
        averaged_predictions, predictions = _partial_dependence_brute(
            model,
            grid_cartesian,
            grid_values,
            feature,
            x,
            method,
            ratio_samples,
        )

        # storing values
        pdp_result = {"grid_values": grid_values}
        if kind == "average":
            pdp_result["average"] = averaged_predictions
        elif kind == "individual":
            pdp_result["individual"] = predictions
        else:
            pdp_result["average"] = averaged_predictions
            pdp_result["individual"] = predictions

        # if there's categorical features, store grid categories
        if isinstance(feature, int):
            is_categories = [is_categorical[feature]]
        else:
            is_categories = [is_categorical[f] for f in feature]
        has_categories = max(is_categories) is True

        if has_categories == 1:
            grid_categories = []
            for i, (is_category, grid_values_) in enumerate(
                zip(is_categories, grid_values)
            ):
                if is_category:
                    if categories_map is not None:
                        if isinstance(feature, int):
                            f = feature
                        else:
                            f = feature[i]
                        grid_categories_ = [categories_map[f][i] for i in grid_values_]
                    else:
                        grid_categories_ = [value for value in grid_values_]
                else:
                    grid_categories_ = []

                grid_categories_ = np.array(grid_categories_)
                grid_categories.append(grid_categories_)
            pdp_result["grid_categories"] = grid_categories

        pdp_results.append(pdp_result)

    return pdp_results


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Args:
        arrays (list): 1-D arrays to form the cartesian product of.
        out: Array to place the cartesian product in.

    Returns:
    out (numpy.ndarray): 2-D array containing the cartesian products formed of input arrays.

    Notes
    -----
    This function may not be used on more than 32 arrays
    because the underlying numpy functions do not support it.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = [len(x) for x in arrays]

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        dtype = np.result_type(*arrays)  # find the most permissive dtype
        if dtype.str[:2] != "<U":
            out = np.empty_like(ix, dtype=dtype)
        else:
            out = np.empty_like(ix, dtype="object")

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def non_cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]

    ref = np.zeros((len(arrays[0]), len(arrays)))
    if out is None:
        out = np.empty_like(ref, dtype="object")

    for n, array in enumerate(arrays):
        out[:, n] = array
    return out
