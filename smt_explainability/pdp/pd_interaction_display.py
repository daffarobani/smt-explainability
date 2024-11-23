from .pd_interaction import pd_overall_interaction, pd_pairwise_interaction
import numpy as np


class PDFeatureInteractionDisplay:
    """
    A class to display feature interactions based on partial dependence.

    Attributes:
        h_scores (list): List of computed H-scores for feature interactions.
        feature_names (list): List of feature names.
    """
    def __init__(self, h_scores, feature_names):
        """
        Initializes the PDFeatureInteractionDisplay class with the given parameters.

        Args:
            h_scores (list): List of computed H-scores for feature interactions.
            feature_names (list): List of feature names.
        """
        self.h_scores = h_scores
        self.feature_names = feature_names

    @classmethod
    def overall_interaction(
        cls,
        model,
        x,
        *,
        features=None,
        categorical_feature_indices=None,
        feature_names=None,
        ratio_samples=None,
    ):
        """
        Creates a PDFeatureInteractionDisplay instance for overall interaction scores.

        Args:
            model (object): The model used for predictions.
            x (numpy.ndarray): Data used for partial dependence computation.
            features (list, optional): List of feature indices.
            categorical_feature_indices (list, optional): Indices of categorical features.
            feature_names (list, optional): List of feature names.
            ratio_samples (float, optional): Ratio of samples to use for computing partial dependence.

        Returns:
            PDFeatureInteractionDisplay: An instance of PDFeatureInteractionDisplay.
        """
        if features is None:
            features = [i for i in range(x.shape[1])]

        if feature_names is None:
            feature_names = [rf"$x_{i}$" for i in range(x.shape[1])]

        h_scores = pd_overall_interaction(
            features,
            x,
            model,
            categorical_feature_indices=categorical_feature_indices,
            ratio_samples=ratio_samples,
        )

        display = PDFeatureInteractionDisplay(h_scores, feature_names)
        return display

    @classmethod
    def pairwise_interaction(
        cls,
        model,
        x,
        feature_pairs,
        *,
        categorical_feature_indices=None,
        feature_names=None,
        ratio_samples=None,
    ):
        """
        Creates a PDFeatureInteractionDisplay instance for pairwise interaction scores.

        Args:
            model (object): The model used for predictions.
            x (numpy.ndarray): Data used for partial dependence computation.
            feature_pairs (list of tuples): List of feature pairs for which interaction scores are computed.
            categorical_feature_indices (list, optional): Indices of categorical features.
            feature_names (list, optional): List of feature names.
            ratio_samples (float, optional): Ratio of samples to use for computing partial dependence.

        Returns:
            PDFeatureInteractionDisplay: An instance of PDFeatureInteractionDisplay.
        """
        if feature_names is None:
            feature_names = [rf"$x_{i}$" for i in range(x.shape[1])]
        interaction_feature_names = list()
        for feature_pair in feature_pairs:
            name = feature_names[feature_pair[0]] + "-" + feature_names[feature_pair[1]]
            interaction_feature_names.append(name)

        h_scores = pd_pairwise_interaction(
            feature_pairs,
            x,
            model,
            categorical_feature_indices=categorical_feature_indices,
            ratio_samples=ratio_samples,
        )

        display = PDFeatureInteractionDisplay(
            h_scores,
            interaction_feature_names,
        )

        return display

    def plot(self, *, figsize=None, sort=False, vert=False):
        """
        Plots the feature interaction scores.

        Args:
            figsize (tuple, optional): Size of the figure.
            sort (bool, optional): Whether to sort the features by interaction scores.
            vert (bool, optional): Whether to plot the bars vertically.

        Returns:
            matplotlib.figure.Figure: The generated plot.
        """

        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": "cmr10",
                "axes.formatter.use_mathtext": True,
            }
        )

        if figsize is None:
            length = max(5, int(len(self.h_scores) * 0.6))
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        feature_names = np.array(self.feature_names)
        h_scores = np.array(self.h_scores)

        if sort:
            vis_feature_names = feature_names[np.argsort(h_scores * -1)]
            vis_h_scores = h_scores[np.argsort(h_scores * -1)]
        else:
            vis_feature_names = feature_names
            vis_h_scores = h_scores

        indexes = np.arange(len(vis_h_scores))
        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        if vert:
            ax.barh(
                indexes, vis_h_scores, color="blue", edgecolor="black", linewidth=0.8
            )
            ax.set_yticks(indexes)
            ax.set_yticklabels(vis_feature_names, fontsize=14)
            ax.set_xlabel("H Score", fontsize=14)
            ax.xaxis.set_tick_params(labelsize=14)
            ax.invert_yaxis()
        else:
            ax.bar(
                indexes, vis_h_scores, color="blue", edgecolor="black", linewidth=0.8
            )
            ax.set_xticks(indexes)
            ax.set_xticklabels(vis_feature_names, fontsize=14)
            ax.set_ylabel("H Score", fontsize=14)
            ax.yaxis.set_tick_params(labelsize=14)

        ax.grid(color="black", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()

        return fig
