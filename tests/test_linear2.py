from copy import deepcopy

import numpy as np
import numpy.testing as nptst
import pytest
from pytest_lazy_fixtures import lf
from scipy.sparse import csr_matrix
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.utils import estimator_checks
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.estimator_checks import parametrize_with_checks

from glmnet import ElasticNet
from tests.util import sanity_check_regression


@pytest.fixture
def min_acceptable_correlation():
    return 0.90


@pytest.fixture
def x_y():
    np.random.seed(488881)
    return make_regression(n_samples=1000, random_state=561)


@pytest.fixture
def x(x_y):
    return x_y[0]


@pytest.fixture
def y(x_y):
    return x_y[1]

@pytest.fixture
def x_sparse(x_y):
    return csr_matrix(x_y[0])


@pytest.fixture
def x_y_wide():
    return make_regression(n_samples=100, n_features=150, random_state=1105)


@pytest.fixture
def x_wide(x_y_wide):
    return x_y_wide[0]


@pytest.fixture
def y_wide(x_y_wide):
    return x_y_wide[1]


@pytest.fixture
def x_wide_sparse(x_wide):
    return csr_matrix(x_wide)


@pytest.fixture(
    params=[
        (lf("x"), lf("y")),
        (lf("x_sparse"), lf("y")),
        (lf("x_wide"), lf("y_wide")),
        (lf("x_wide_sparse"), lf("y_wide")),
    ]
)
def x_y_inputs(request):
    return request.param


@pytest.fixture(
    params=[
        (lf("x"), lf("y")),
        (lf("x_sparse"), lf("y")),
    ]
)
def x_y_tall_inputs(request):
    return request.param


@pytest.fixture(
    params=[
        (lf("x_wide"), lf("y_wide")),
        (lf("x_wide_sparse"), lf("y_wide")),
    ]
)
def x_y_wide_inputs(request):
    return request.param


@pytest.fixture(
    params=[0.0, 0.25, 0.50, 0.75, 1.0]
)
def alphas(request):
    return request.param


@pytest.fixture
def n_splits():
    return [-1, 0, 5]

# NOT creating a lot of models with specific seeds
# if it is important, we can try changing the seed
# per-func
@pytest.fixture
def m():
    return ElasticNet()


@pytest.fixture(params=[0.0, 0.25, 0.50, 0.75, 1.0])
def malphas(alpha):
    return ElasticNet(alpha=alpha.param, random_state=2465)


@pytest.fixture(params=[-1, 0, 5])
def mnsplits(splits):
    return ElasticNet(n_splits=splits.params, random_state=6601)


@pytest.fixture
def scoring():
    return [
            "r2",
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_error",
        ]

# @parametrize_with_checks([ElasticNet()])
# def test_sklearn_compatible_estimator(estimator, check):
#     check(estimator)


@pytest.mark.parametrize("inputs", [(lf("x_y_inputs"))])
def test_with_defaults(m, inputs):
    # print(f"{meta_inputs=}")
    x, y = inputs
    m = m.fit(x, y)
    sanity_check_regression(m, x)

    # check selection of lambda_best
    assert m.lambda_best_inx_ <= m.lambda_max_inx_

    # check full path predict
    p = m.predict(x, lamb=m.lambda_path_)
    assert p.shape[-1] == m.lambda_path_.size


@pytest.mark.parametrize("inputs", [(lf("x_y_inputs"))])
def test_one_row_predict(m, inputs):
    # Verify that predicting on one row gives only one row of output
    X, y = inputs
    m.fit(X, y)
    p = m.predict(X[0].reshape((1, -1)))
    assert p.shape == (1,)


@pytest.mark.parametrize("inputs", [(lf("x_y_inputs"))])
def test_one_row_predict_with_lambda(m, inputs):
    # One row to predict along with lambdas should give 2D output
    X, y = inputs
    m.fit(X, y)
    p = m.predict(X[0].reshape((1, -1)), lamb=[20, 10])
    assert p.shape == (1, 2)


def test_with_single_var(m, min_acceptable_correlation):
    x = np.random.rand(500, 1)
    y = (1.3 * x).ravel()

    m = m.fit(x, y)
    score = r2_score(y, m.predict(x))
    assert score >= min_acceptable_correlation


def test_with_no_predictor_variance(m):
    x = np.ones((500, 1))
    y = np.random.rand(500)

    with pytest.raises(ValueError, match=r".*7777.*"):
        m.fit(x, y)


@pytest.mark.parametrize("inputs", [(lf("x_y_inputs"))])
def test_relative_penalties(m, inputs):
    x, y = inputs
    m1 = m
    m2 = deepcopy(m1)
    p = x.shape[1]

    # m1 no relative penalties applied
    m1.fit(x, y)

    # find the nonzero indices from LASSO
    nonzero = np.nonzero(m1.coef_)

    # unpenalize those nonzero coefs
    penalty = np.repeat(1, p)
    penalty[nonzero] = 0

    # refit the model with the unpenalized coefs
    m2.fit(x, y, relative_penalties=penalty)

    # verify that the unpenalized coef ests exceed the penalized ones
    # in absolute value
    assert np.all(np.abs(m1.coef_) <= np.abs(m2.coef_))


@pytest.mark.parametrize("inputs,alpha", [(lf("x_y_inputs"), lf("alphas"))])
def test_alphas(m, inputs, alpha, min_acceptable_correlation):
    x, y = inputs
    m.alpha = alpha
    m = m.fit(x, y)
    score = r2_score(y, m.predict(x))
    assert score >= min_acceptable_correlation


@pytest.fixture(params=[1000,1000,100,100])
def n_samples(request):
    return request.param

@pytest.fixture
def m_with_limits(x):
    return ElasticNet(
        lower_limits=np.repeat(-1, x.shape[1]),
        upper_limits=0,
        alpha=0
    )


@pytest.fixture
def m_with_limits_wide(x_wide):
    return ElasticNet(
        lower_limits=np.repeat(-1, x_wide.shape[1]),
        upper_limits=0,
        alpha=0
    )

# TODO I think it should be possible to merge the tall and wide
# tests here, I just haven't figured exactly how yet
@pytest.mark.parametrize("inputs", [(lf("x_y_tall_inputs"))])
def test_coef_tall_limits(m_with_limits, inputs):
    x, y=inputs
    m_with_limits = m_with_limits.fit(x, y)
    assert np.all(m_with_limits.coef_ >= -1)
    assert np.all(m_with_limits.coef_ <= 0)


@pytest.mark.parametrize("inputs", [(lf("x_y_wide_inputs"))])
def test_coef_wide_limits(m_with_limits_wide, inputs):
    x, y=inputs
    m_with_limits_wide = m_with_limits_wide.fit(x, y)
    assert np.all(m_with_limits_wide.coef_ >= -1)
    assert np.all(m_with_limits_wide.coef_ <= 0)
