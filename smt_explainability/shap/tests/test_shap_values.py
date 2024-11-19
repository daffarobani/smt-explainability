from smt.utils.sm_test_case import SMTestCase
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KRG,
    KPLS,
    MixIntKernelType,
    MixHrcKernelType,
)
from smt.applications.mixed_integer import MixedIntegerKrigingModel
from smt.problems import WingWeight
from smt_explainability.problems import MixedCantileverBeam
from smt_explainability.shap import compute_shap_values

import numpy as np
import unittest


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPartialDependenceNumerical(SMTestCase):
    def setUp(self):
        nsamples = 100
        n_train = int(0.8 * nsamples)
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
        x = sampling(nsamples)
        y = fun(x)
        x_tr, _y_tr = x[:n_train, :], y[:n_train]
        x_te, _y_te = x[n_train:, :], y[n_train:]
        is_categorical = [False] * x.shape[1]

        sm = KRG(
            theta0=[1e-2] * x.shape[1],
            print_prediction=False
        )
        sm.set_training_values(x_tr, _y_tr)
        sm.train()

        self.model = sm
        self.x_tr = x_tr
        self.x_te = x_te
        self.nsamples = nsamples
        self.is_categorical = is_categorical

    def test_kernel_shap(self):
        shap_values = compute_shap_values(
            self.model,
            self.x_te,
            self.x_tr,
            self.is_categorical,
            method="kernel",
        )
        assert shap_values.shape == (self.x_te.shape[0], self.x_te.shape[1])

    def test_exact_shap(self):
        shap_values = compute_shap_values(
            self.model,
            self.x_te,
            self.x_tr,
            self.is_categorical,
            method="exact",
        )
        assert shap_values.shape == (self.x_te.shape[0], self.x_te.shape[1])


class TestPartialDependenceMixed(SMTestCase):
    def setUp(self):
        nsamples = 50
        n_train = int(0.8 * nsamples)

        fun = MixedCantileverBeam()
        ds = DesignSpace(
            [
                CategoricalVariable(values=[str(i + 1) for i in range(12)]),
                FloatVariable(10.0, 20.0),
                FloatVariable(1.0, 2.0),
            ]
        )
        x = fun.sample(nsamples)
        y = fun(x)
        x_tr, _y_tr = x[:n_train, :], y[:n_train]
        x_te, _y_te = x[n_train:, :], y[n_train:]

        # Index for categorical features
        categorical_feature_indices = [0]
        is_categorical = [False] * x.shape[1]
        for feature_idx in categorical_feature_indices:
            is_categorical[feature_idx] = True

        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                design_space=ds,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ARC_KERNEL,
                theta0=np.array([4.43799547e-04, 4.39993134e-01, 1.59631650e+00]),
                corr="squar_exp",
                n_start=1,
                cat_kernel_comps=[2],
                n_comp=2,
                print_global=False,
            ),
        )
        sm.set_training_values(x, np.array(y))
        sm.train()

        self.model = sm
        self.x_tr = x_tr
        self.x_te = x_te
        self.nsamples = nsamples
        self.is_categorical = is_categorical

    def test_kernel_shap(self):
        shap_values = compute_shap_values(
            self.model,
            self.x_te,
            self.x_tr,
            self.is_categorical,
            method="kernel",
        )
        assert shap_values.shape == (self.x_te.shape[0], self.x_te.shape[1])

    def test_exact_shap(self):
        shap_values = compute_shap_values(
            self.model,
            self.x_te,
            self.x_tr,
            self.is_categorical,
            method="exact",
        )
        assert shap_values.shape == (self.x_te.shape[0], self.x_te.shape[1])


if __name__ == "__main__":
    unittest.main()
