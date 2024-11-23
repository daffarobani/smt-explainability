import numpy as np
from copy import deepcopy
from sobolsampling.sobol_new import sobol_points


class SobolIndices:
    """
    A class to calculate Sobol sensitivity indices.

    Attributes:
        nvar (int): Number of variables.
        model (object): The model used for predictions.
        n (int): Number of Monte Carlo samples.
        A (numpy.ndarray): Sample matrix A.
        B (numpy.ndarray): Sample matrix B.
        ya (numpy.ndarray): Model predictions for A.
        yb (numpy.ndarray): Model predictions for B.
        fo_2 (float): Mean of the squared model predictions.
        denom (float): Denominator for Sobol index calculations.
    """

    def __init__(
            self,
            nvar: int,
            model,
            *,
            x_bounds=None,
            x=None,
            percentiles=(0.05, 0.95),
            n_mc=2e5
    ):
        """
        Initializes the SobolIndices class with the given parameters.

        Args:
            nvar (int): Number of variables.
            model: The model used for predictions.
            x_bounds (numpy.ndarray, optional): Bounds for the variables.
            x (numpy.ndarray, optional): Data to calculate percentiles for bounds.
            percentiles (tuple, optional): Percentiles to calculate bounds.
            n_mc (float, optional): Number of Monte Carlo samples.
        """

        self.nvar = nvar
        self.model = model
        self.n = int(n_mc)

        if x_bounds is None:
            x_bounds = np.zeros(shape=[nvar, 2])
            x_bounds[:, 0] = np.percentile(x, percentiles[0]*100, axis=0)
            x_bounds[:, 1] = np.percentile(x, percentiles[1]*100, axis=0)

        bounds = np.concatenate([x_bounds, x_bounds], axis=0)
        lb = bounds[:, 0]
        ub = bounds[:, 1]

        x_sample = sobol_sampling(self.n, self.nvar * 2, lb, ub)

        self.A = x_sample[:, :self.nvar]
        self.B = x_sample[:, self.nvar:]
        del x_sample

        self.ya = None
        self.yb = None
        self.fo_2 = None
        self.denom = None

    def analyze(self, first_order=True, total_order=False, second_order=False):
        """
        Analyzes the Sobol indices.

        Args:
            first_order (bool, optional): Whether to calculate first order indices.
            total_order (bool, optional): Whether to calculate total order indices.
            second_order (bool, optional): Whether to calculate second order indices.

        Returns:
            dict: A dictionary containing the calculated indices.
        """

        n_samples = self.A.shape[0]

        ya = np.zeros(shape=[n_samples, 1])

        if n_samples < 10000:
            ya = self.model.predict_values(self.A)
        else:
            run_times = int(np.ceil(n_samples / 10000))
            for i in range(run_times):
                start = i * 10000
                stop = (i + 1) * 10000
                if i != (run_times - 1):
                    ya[start: stop, :] = self.model.predict_values(self.A[start: stop, :])
                else:
                    ya[start:, :] = self.model.predict_values(self.A[start:, :])

        yb = np.zeros(shape=[n_samples, 1])

        if n_samples < 10000:
            yb = self.model.predict_values(self.B)
        else:
            run_times = int(np.ceil(n_samples / 10000))
            for i in range(run_times):
                start = i * 10000
                stop = (i + 1) * 10000
                if i != (run_times - 1):
                    yb[start: stop, :] = self.model.predict_values(self.B[start: stop, :])
                else:
                    yb[start:, :] = self.model.predict_values(self.B[start:, :])

        fo_2 = (np.sum(ya) / self.n) ** 2
        denom = (np.sum(ya ** 2) / self.n) - fo_2

        self.ya = ya
        self.yb = yb
        self.fo_2 = fo_2
        self.denom = denom

        indices = dict()
        if first_order or total_order:
            first_indices, total_indices = self.calc_ft_order(first_order, total_order)
            if first_order:
                indices['first'] = first_indices
            if total_order:
                indices['total'] = total_indices

        if second_order:
            if not first_order:
                first_indices, _ = self.calc_ft_order(True, False)
            else:
                first_indices = indices['first']
            indices["second"] = self.calc_second_order(first_indices)

        return indices

    def calc_ft_order(self, first=True, total=False):
        """
        Calculate first and total order Sobol Indices.

        Args:
            first (bool, optional): Whether to calculate first order indices.
            total (bool, optional): Whether to calculate total order indices.

        Returns:
            list: A list containing first and total order indices.
        """

        s1 = np.zeros(self.nvar)
        st = np.zeros(self.nvar)

        for ii in range(self.nvar):
            c_i = deepcopy(self.B)
            c_i[:, ii] = self.A[:, ii]

            # Use model to predict Monte-Carlo
            nsamp = c_i.shape[0]
            yci = np.zeros(shape=[nsamp, 1])
            if nsamp <= 10000:
                yci = self.model.predict_values(c_i)
            else:
                run_times = int(np.ceil(nsamp / 10000))
                for i in range(run_times):
                    start = i * 10000
                    stop = (i + 1) * 10000
                    if i != (run_times - 1):
                        yci[start: stop, :] = self.model.predict_values(c_i[start: stop, :])
                    else:
                        yci[start:, :] = self.model.predict_values(c_i[start:, :])

            if first:
                s1[ii] = ((1 / self.n) * np.sum(self.ya * yci) - self.fo_2) / self.denom
            if total:
                st[ii] = 1 - (((1 / self.n) * np.sum(self.yb * yci) - self.fo_2) / self.denom)

        return [s1, st]

    def calc_second_order(self, s1):
        """
        Calculate second order indices.

        Args:
        s1 (numpy.ndarray): First order indices.

        Returns:
        dict: A dictionary containing second order indices.
        """

        s2 = dict()

        for ii in range(self.nvar - 1):
            for jj in range(ii + 1, self.nvar):
                c_ij = deepcopy(self.B)
                c_ij[:, ii] = self.A[:, ii]
                c_ij[:, jj] = self.A[:, jj]

                nsamp = c_ij.shape[0]
                yci = np.zeros(shape=[nsamp, 1])
                if nsamp <= 10000:
                    yci = self.model.predict_values(c_ij)
                else:
                    run_times = int(np.ceil(nsamp / 10000))
                    for i in range(run_times):
                        start = i * 10000
                        stop = (i + 1) * 10000
                        if i != (run_times - 1):
                            yci[start: stop, :] = self.model.predict_values(c_ij[start:stop, :])
                        else:
                            yci[start:, :] = self.model.predict_values(c_ij[start:, :])

                vij = ((1 / self.n) * np.sum(self.ya * yci) - self.fo_2)
                key = "x" + str(ii) + "-x" + str(jj)
                s2[key] = (vij / self.denom) - s1[ii] - s1[jj]

        return s2


def sobol_sampling(n_samples, n_var, lb, ub):
    """
    Generates Sobol samples within the given bounds.

    Args:
        n_samples (int): Number of samples.
        n_var (int): Number of variables.
        lb (numpy.ndarray): Lower bounds for the variables.
        ub (numpy.ndarray): Upper bounds for the variables.

    Returns:
        numpy.ndarray: Generated Sobol samples.
    """

    samplenorm = sobol_points(n_samples+1, n_var)
    samplenorm = samplenorm[1:, :]

    samples = np.zeros(shape=[n_samples, n_var])
    for i in range(0, n_var):
        for j in range(0, n_samples):
            samples[j, i] = (samplenorm[j, i] * (ub[i] - lb[i])) + lb[i]
    return samples
