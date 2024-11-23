from .sobol_indices import SobolIndices
import numpy as np


class SobolIndicesDisplay:
    """
    A class to display Sobol sensitivity indices.

    Attributes:
        sobol_indices (dict): Sobol indices.
        feature_names (list): Names of the features.
    """
    def __init__(self, nvar, sobol_indices, feature_names):
        """
        Initializes the SobolIndicesDisplay class with the given parameters.

        Args:
            nvar (int): Number of variables.
            sobol_indices (dict): Calculated Sobol indices.
            feature_names (list, optional): Names of the features.
        """
        self.sobol_indices = sobol_indices

        if feature_names is None:
            feature_names = [rf"$x_{i}$" for i in range(nvar)]
        self.feature_names = feature_names

    @classmethod
    def from_surrogate_model(
        cls,
        nvar,
        model,
        *,
        first_order=True,
        total_order=True,
        second_order=False,
        x_bounds=None,
        x=None,
        percentiles=(0.05, 0.95),
        n_mc=2e5,
        feature_names=None,
    ):
        """
        Creates a SobolIndicesDisplay instance from a surrogate model.

        Args:
            nvar (int): Number of variables.
            model (object): The surrogate model used for predictions.
            first_order (bool, optional): Whether to calculate first order indices.
            total_order (bool, optional): Whether to calculate total order indices.
            second_order (bool, optional): Whether to calculate second order indices.
            x_bounds (numpy.ndarray, optional): Bounds for the variables.
            x (numpy.ndarray, optional): Data to calculate percentiles for bounds.
            percentiles (tuple, optional): Percentiles to calculate bounds.
            n_mc (float, optional): Number of Monte Carlo samples.
            feature_names (list, optional): Names of the features.

        Returns:
            SobolIndicesDisplay: An instance of SobolIndicesDisplay.
        """

        sobol_indices = SobolIndices(
            nvar,
            model,
            x_bounds=x_bounds,
            x=x,
            percentiles=percentiles,
            n_mc=n_mc
        ).analyze(first_order, total_order, second_order)
        display = SobolIndicesDisplay(nvar, sobol_indices, feature_names)
        return display

    def plot(self, order, *, sort=False, figsize=None, max_num_display=None):
        """
        Plots the Sobol indices.

        Args:
            order (str): The order of indices to plot ('first', 'total', or 'second').
            sort (bool, optional): Whether to sort based on the Sobol indices.
            figsize (tuple, optional): Size of the figure.
            max_num_display (int, optional): Maximum number of indices to display.

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

        if order in ["first", "total"]:
            indices = self.feature_names
            values = self.sobol_indices[order]
        elif order == "second":
            indices = list(self.sobol_indices['second'].keys())
            for i in range(len(indices)):
                feature_i, feature_j = indices[i].split("-")
                feature_i = self.feature_names[int(feature_i.replace('x', ''))]
                feature_j = self.feature_names[int(feature_j.replace('x', ''))]
                indices[i] = rf"{feature_i}-{feature_j}"

            indices = np.array(indices)
            values = np.array(list(self.sobol_indices['second'].values()))
        else:
            raise ValueError("order must be either 'first', 'total', or 'second'.")

        if sort:
            indices = indices[np.argsort(values*-1)]
            values = values[np.argsort(values*-1)]

        if max_num_display is not None:
            if max_num_display < len(indices):
                vis_indices = indices[:max_num_display]
                vis_values = values[:max_num_display]
                vis_indices = np.append(vis_indices, "Others")
                vis_values = np.append(vis_values, np.sum(values[max_num_display:]))
            else:
                vis_indices = indices
                vis_values = values
        else:
            vis_indices = indices
            vis_values = values

        if figsize is None:
            length = max(5, int(len(vis_values) * 0.6))
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        ax.bar(
            np.arange(len(vis_values)),
            vis_values,
            color="blue",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xticks(np.arange(len(vis_values)))
        ax.set_xticklabels(vis_indices, fontsize=14)
        ax.set_ylabel("Sobol indices", fontsize=14)
        ax.grid(color="black", alpha=0.2)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_axisbelow(True)
        fig.tight_layout()

        # Close the figure before returning
        plt.close(fig)

        return fig
