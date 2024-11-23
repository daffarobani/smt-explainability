from .shap_values import compute_shap_values
import numpy as np


def compute_shap_feature_importance(
    instances,
    model,
    x,
    is_categorical,
    *,
    method="kernel"
):
    """
    Computes feature importance based on SHAP values on the given instances.

    Args:
        instances (numpy.ndarray): Instances for which SHAP values are computed.
        model (object): The model used for predictions.
        x (numpy.ndarray): Reference data used for SHAP value computation.
        is_categorical (list): List indicating whether each feature is categorical.
        method (str, optional): Method to use for SHAP value computation ('kernel' or 'exact').

    Returns:
        numpy.ndarray: Computed SHAP feature importances.
    """
    shap_values = compute_shap_values(
        model,
        instances,
        x,
        is_categorical,
        method=method,
    )
    feature_importances = np.abs(shap_values).mean(axis=0)
    return feature_importances
