from itertools import product
from scipy.special import comb
import numpy as np


def create_mask_array(m):
    mask = np.array(list(product(range(2), repeat=m)))
    # remove mask where all elements are 0 / 1
    mask = mask[
        (~np.all(mask == 0, axis=1)) &
        (~np.all(mask == 1, axis=1))
    ]
    return mask


def calculate_weight(mask_row):
    m = len(mask_row)
    z = np.sum(mask_row)
    numerator = m - 1
    denominator = comb(m, z) * z * (m - z)
    weight = numerator / denominator
    return weight


def compute_shap_values(mask, s_full, weights, reference_values, model):
    y = model.predict_values(s_full)
    # b0 = model.predict_values(reference_values.reshape(1, -1))
    b0 = model.predict_values(reference_values)
    y = y - b0

    w = np.diag(weights)

    b = np.dot(
        np.linalg.inv(np.dot(np.dot(mask.transpose(), w), mask)),
        np.dot(np.dot(mask.transpose(), w), y)
    )
    b = b.reshape(-1, )
    return b


def individual_shap_values(model, x_obs, x_ref, is_categorical, *, method="kernel"):
    # reference_values = get_reference_feature_values(x, is_categorical)
    shap_values = list()

    for x in x_obs:
        instance = x.reshape(1, -1)
        mask = create_mask_array(instance.shape[1])
        # mask = np.repeat(mask, 5, axis=0)
        # s_with_zero = mask * instance
        reference_values = np.ones(mask.shape)
        for i in range(len(reference_values)):
            reference_values[i, :] = get_reference_feature_values(x_ref, is_categorical)
        # s_full = (s_with_zero == 0) * reference_values + s_with_zero
        s_ref = (mask == 0) * reference_values
        s_real = (mask == 1) * instance
        s_full = s_ref + s_real

        weights = np.apply_along_axis(calculate_weight, 1, mask)
        shap_value = compute_shap_values(mask, s_full, weights, reference_values, model)
        shap_values.append(shap_value)
    shap_values = np.array(shap_values)
    return shap_values


def get_reference_feature_values(x, is_categorical):
    # get reference values for each feature
    # if the feature is categorical/ordinal -> random
    # else -> mean
    num_features = x.shape[1]
    reference_values = np.zeros(num_features)
    for feature_idx in range(num_features):
        if is_categorical[feature_idx]:
            # mode = stats.mode(x[:, feature_idx], keepdims=False)[0]
            # reference_values[feature_idx] = mode
            reference_values[feature_idx] = np.random.choice(x[:, feature_idx])
        else:
            mean = np.mean(x[:, feature_idx])
            reference_values[feature_idx] = mean
    return reference_values
