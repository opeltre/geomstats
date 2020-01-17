"""Template unit tests for scikit-learn estimators."""

from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator

import geomstats.backend as gs
import geomstats.tests

from geomstats.learning._template import (TemplateEstimator,
                                          TemplateTransformer,
                                          TemplateClassifier)


ESTIMATORS = (TemplateEstimator, TemplateTransformer, TemplateClassifier)


class TestEstimators(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.data = load_iris(return_X_y=True)

    def test_check_template_estimator(self):
        check_estimator(TemplateEstimator)

    def test_check_template_transformer(self):
        check_estimator(TemplateTransformer)

    def test_check_template_classifier(self):
        check_estimator(TemplateClassifier)

    def test_template_estimator(self):
        est = TemplateEstimator()
        self.assertEqual(est.demo_param, 'demo_param')

        X, y = self.data

        est.fit(X, y)
        self.assertTrue(hasattr(est, 'is_fitted_'))

        y_pred = est.predict(X)
        self.assertAllClose(y_pred, gs.ones(gs.shape(X)[0]))

    def test_template_transformer_error(self):
        X, y = self.data
        n_samples = gs.shape(X)[0]
        trans = TemplateTransformer()
        trans.fit(X)
        X_diff_size = gs.ones((n_samples, gs.shape(X)[1] + 1))
        self.assertRaises(ValueError, trans.transform, X_diff_size)

    def test_template_transformer(self):
        X, y = self.data
        trans = TemplateTransformer()
        assert trans.demo_param == 'demo'

        trans.fit(X)
        assert trans.n_features_ == X.shape[1]

        X_trans = trans.transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))

        X_trans = trans.fit_transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))

    def test_template_classifier(self):
        X, y = self.data
        clf = TemplateClassifier()
        assert clf.demo_param == 'demo'

        clf.fit(X, y)
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'X_')
        assert hasattr(clf, 'y_')

        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],)
