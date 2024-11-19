from smt.utils.sm_test_case import SMTestCase
from smt.sampling_methods import LHS
from smt.problems import WingWeight
from smt_explainability.sobol import SobolIndices
from smt.surrogate_models import KRG

import unittest


class TestSobolIndices(SMTestCase):
    def test_sobol_indices_with_xlimits(self):
        nsamples = 80
        first_order = True
        total_order = True
        second_order = True
        n_mc = 1e5

        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
        x = sampling(nsamples)
        y = fun(x)
        nvar = x.shape[1]

        sm = KRG(
            theta0=[1e-2] * nvar,
            print_prediction=False
        )
        sm.set_training_values(x, y)
        sm.train()

        sobol_indices = SobolIndices(
            nvar, sm, x_bounds=fun.xlimits, n_mc=n_mc).analyze(first_order, total_order, second_order)

        assert len(sobol_indices["first"]) == nvar
        assert len(sobol_indices["total"]) == nvar
        desired_second_order_keys = {f"x{i}-x{j}" for i in range(nvar) for j in range(i+1, nvar)}
        assert set(sobol_indices["second"].keys()) == desired_second_order_keys

    def test_sobol_indices_with_percentiles(self):
        nsamples = 80
        first_order = True
        total_order = True
        second_order = True
        n_mc = 1e5
        percentiles = (0.02, 0.98)

        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
        x = sampling(nsamples)
        y = fun(x)
        nvar = x.shape[1]

        sm = KRG(
            theta0=[1e-2] * nvar,
            print_prediction=False
        )
        sm.set_training_values(x, y)
        sm.train()

        sobol_indices = SobolIndices(
            nvar, sm, x=x, percentiles=percentiles, n_mc=n_mc).analyze(first_order, total_order, second_order)

        assert len(sobol_indices["first"]) == nvar
        assert len(sobol_indices["total"]) == nvar
        desired_second_order_keys = {f"x{i}-x{j}" for i in range(nvar) for j in range(i + 1, nvar)}
        assert set(sobol_indices["second"].keys()) == desired_second_order_keys


if __name__ == "__main__":
    unittest.main()
