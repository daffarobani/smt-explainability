from smt.utils.sm_test_case import SMTestCase
from smt.sampling_methods import LHS
from smt.problems import WingWeight
from smt_ex.sobol import SobolIndicesDisplay
from smt.surrogate_models import KRG

import unittest


class TestSobolIndicesDisplay(SMTestCase):
    def test_sobol_indices_display_with_xlimits(self):
        nsamples = 80
        first_order = True
        total_order = True
        second_order = True
        n_mc = 1e5
        feature_names = [
            r"$S_{w}$",
            r"$W_{fw}$",
            r"$A$",
            r"$\Delta$",
            r"$q$",
            r"$\lambda$",
            r"$t_{c}$",
            r"$N_{z}$",
            r"$W_{dg}$",
            r"$W_{p}$",
        ]

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

        sobol_display = SobolIndicesDisplay.from_surrogate_model(
            nvar,
            sm,
            x_bounds=fun.xlimits,
            n_mc=n_mc,
            first_order=first_order,
            total_order=total_order,
            second_order=second_order,
            feature_names=feature_names,
        )

        sobol_display.plot('first')
        sobol_display.plot('total')
        sobol_display.plot('second')

        assert len(sobol_display.sobol_indices["first"]) == nvar
        assert len(sobol_display.sobol_indices["total"]) == nvar
        desired_second_order_keys = {f"x{i}-x{j}" for i in range(nvar) for j in range(i+1, nvar)}
        assert set(sobol_display.sobol_indices["second"].keys()) == desired_second_order_keys

    def test_sobol_indices_with_percentiles(self):
        nsamples = 80
        first_order = True
        total_order = True
        second_order = True
        percentiles = (0.02, 0.98)
        n_mc = 1e5
        feature_names = [
            r"$S_{w}$",
            r"$W_{fw}$",
            r"$A$",
            r"$\Delta$",
            r"$q$",
            r"$\lambda$",
            r"$t_{c}$",
            r"$N_{z}$",
            r"$W_{dg}$",
            r"$W_{p}$",
        ]

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

        sobol_display = SobolIndicesDisplay.from_surrogate_model(
            nvar,
            sm,
            x=x,
            percentiles=percentiles,
            n_mc=n_mc,
            first_order=first_order,
            total_order=total_order,
            second_order=second_order,
            feature_names=feature_names,
        )

        sobol_display.plot('first')
        sobol_display.plot('total')
        sobol_display.plot('second')

        assert len(sobol_display.sobol_indices["first"]) == nvar
        assert len(sobol_display.sobol_indices["total"]) == nvar
        desired_second_order_keys = {f"x{i}-x{j}" for i in range(nvar) for j in range(i + 1, nvar)}
        assert set(sobol_display.sobol_indices["second"].keys()) == desired_second_order_keys


if __name__ == "__main__":
    unittest.main()
