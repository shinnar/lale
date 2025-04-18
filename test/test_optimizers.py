# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportMissingImports=false

import tempfile
import unittest
import warnings
from test import EnableSchemaValidation

import jsonschema
import numpy as np
import pandas as pd
from packaging import version
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split

from lale import schemas
from lale.helpers import get_sklearn_estimator_name, with_fixed_estimator_name
from lale.lib.lale import (
    SMAC,
    ConcatFeatures,
    GridSearchCV,
    Hyperopt,
    NoOp,
    OptimizeLast,
    Project,
    TopKVotingClassifier,
)
from lale.lib.rasl.ordinal_encoder import OrdinalEncoder
from lale.lib.sklearn import (
    PCA,
    SVC,
    KNeighborsClassifier,
    KNeighborsRegressor,
    LinearRegression,
    LogisticRegression,
    MinMaxScaler,
    Normalizer,
    Nystroem,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    SimpleImputer,
    StandardScaler,
    TfidfVectorizer,
)
from lale.operators import TrainedIndividualOp, TrainedPipeline, sklearn_version
from lale.search.op2hp import hyperopt_search_space

from .mock_custom_operators import OpThatWorksWithFiles


def f_min(op, X, y, num_folds=5):
    from lale.helpers import cross_val_score

    # try:
    scores = cross_val_score(op, X, y, cv=num_folds)

    return 1 - np.mean(scores)  # Minimize!
    # except BaseException as e:
    #     print(e)
    #     return


def iris_f_min(op, num_folds=5):
    from sklearn import datasets

    iris = datasets.load_iris()
    return f_min(op, iris.data, iris.target, num_folds=num_folds)


def iris_f_min_for_folds(num_folds=5):
    return lambda op: iris_f_min(op, num_folds=num_folds)


def iris_fmin_tae(op, num_folds=5):
    from lale.search.lale_smac import lale_op_smac_tae

    return lale_op_smac_tae(op, iris_f_min_for_folds(num_folds=num_folds))


@unittest.skip(
    "SMAC interface is not currently supported with newer version of Python",
)
class TestSMAC(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_smac(self):
        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace
        from smac.facade.smac_facade import SMAC as orig_SMAC
        from smac.scenario.scenario import Scenario

        from lale.search.lale_smac import get_smac_space

        # Import SMAC-utilities
        lr = LogisticRegression()

        cs: ConfigurationSpace = get_smac_space(lr)

        # Scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": 1,  # maximum function evaluations
                "cs": cs,  # configuration space
                "deterministic": "true",
                "abort_on_first_run_crash": False,
            }
        )

        # Optimize, using a SMAC-object
        tae = iris_fmin_tae(lr, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = orig_SMAC(
            scenario=scenario, rng=np.random.RandomState(42), tae_runner=tae
        )

        incumbent = smac.optimize()

        inc_value = tae(incumbent)

        print(f"Optimized Value: {inc_value:.2f}")

    def dont_test_smac_choice(self):
        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace
        from smac.facade.smac_facade import SMAC as orig_SMAC
        from smac.scenario.scenario import Scenario

        from lale.search.lale_smac import get_smac_space

        # Import SMAC-utilities

        tfm = PCA() | Nystroem() | NoOp()
        if sklearn_version >= version.Version("1.2"):
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        planned_pipeline1 = (
            (ohe | NoOp()) >> tfm >> (LogisticRegression() | KNeighborsClassifier())
        )

        cs: ConfigurationSpace = get_smac_space(planned_pipeline1, lale_num_grids=1)

        # Scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": 1,  # maximum function evaluations
                "cs": cs,  # configuration space
                "deterministic": "true",
            }
        )

        # Optimize, using a SMAC-object
        tae = iris_fmin_tae(planned_pipeline1, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = orig_SMAC(
            scenario=scenario, rng=np.random.RandomState(42), tae_runner=tae
        )

        incumbent = smac.optimize()

        inc_value = tae(incumbent)

        print(f"Optimized Value: {inc_value:.2f}")

    def test_smac1(self):
        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        opt = SMAC(estimator=planned_pipeline, max_evals=1)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        _ = res.predict(self.X_test)

    def test_smac2(self):
        from test.mock_module import BadClassifier

        import lale.operators

        BadClf = lale.operators.make_operator(BadClassifier)
        planned_pipeline = (PCA | NoOp) >> BadClf()
        opt = SMAC(estimator=planned_pipeline, max_evals=1)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        # Get the trials object and make sure that SMAC assigned cost_for_crash which is MAXINT by default to
        # at least one trial (correspond to KNN).
        trials = res._impl.get_trials()
        assert 2147483647.0 in trials.cost_per_config.values()

    def test_smac_timeout_zero_classification(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )
        opt = SMAC(estimator=planned_pipeline, max_evals=1, max_opt_time=0.0)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        assert res.get_pipeline() is None

    def test_smac_timeout_zero_regression(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)
        opt = SMAC(
            estimator=planned_pipeline, scoring="r2", max_evals=1, max_opt_time=0.0
        )
        # run optimizer
        res = opt.fit(X[:500, :], y[:500])
        assert res.get_pipeline() is None

    def test_smac_timeout_classification(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )
        max_opt_time = 4.0
        opt = SMAC(estimator=planned_pipeline, max_evals=1, max_opt_time=max_opt_time)

        start = time.time()
        _ = opt.fit(self.X_train, self.y_train)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 1.2
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"

    def test_smac_timeout_regression(self):
        import time

        from lale.datasets.util import load_boston

        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        X, y = load_boston(return_X_y=True)
        max_opt_time = 2.0
        opt = SMAC(
            estimator=planned_pipeline,
            scoring="r2",
            max_evals=1,
            max_opt_time=max_opt_time,
        )

        start = time.time()
        _ = opt.fit(X[:500, :], y[:500])
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.5
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"


def run_hyperopt_on_planned_pipeline(planned_pipeline, max_iters=1):
    # data
    features, labels = load_iris(return_X_y=True)
    # set up optimizer

    opt = Hyperopt(estimator=planned_pipeline, max_evals=max_iters)
    # run optimizer
    _ = opt.fit(features, labels)


class TestVisitorErrors(unittest.TestCase):
    def test_empty_schema(self):
        pca = PCA().customize_schema(whiten=schemas.Schema())
        plan = (
            (pca & (MinMaxScaler | Normalizer))
            >> ConcatFeatures()
            >> (MinMaxScaler | Normalizer)
            >> (LogisticRegression | KNeighborsClassifier)
        )
        from lale.search.schema2search_space import OperatorSchemaError

        with self.assertRaises(OperatorSchemaError):
            run_hyperopt_on_planned_pipeline(plan)

    #        print(str(ctxt.exception))

    def test_no_max_schema(self):
        pca = PCA().customize_schema(n_components=schemas.Float(minimum=0.0))
        plan = (
            (pca & (MinMaxScaler | Normalizer))
            >> ConcatFeatures()
            >> (MinMaxScaler | Normalizer)
            >> (LogisticRegression | KNeighborsClassifier)
        )
        from lale.search.search_space import SearchSpaceError

        with self.assertRaises(SearchSpaceError):
            run_hyperopt_on_planned_pipeline(plan)


#        print(str(ctxt.exception))


class TestHyperoptOperatorDuplication(unittest.TestCase):
    def test_planned_pipeline_1(self):
        plan = (
            (PCA & (MinMaxScaler | Normalizer))
            >> ConcatFeatures()
            >> (MinMaxScaler | Normalizer)
            >> (LogisticRegression | KNeighborsClassifier)
        )
        run_hyperopt_on_planned_pipeline(plan)

    def test_planned_pipeline_2(self):
        plan = (
            (MinMaxScaler() & NoOp())
            >> ConcatFeatures()
            >> (NoOp() & MinMaxScaler())
            >> ConcatFeatures()
            >> (LogisticRegression | KNeighborsClassifier)
        )
        run_hyperopt_on_planned_pipeline(plan)

    def test_planned_pipeline_3(self):
        plan = (
            (MinMaxScaler() & NoOp())
            >> ConcatFeatures()
            >> (StandardScaler & (NoOp() | MinMaxScaler()))
            >> ConcatFeatures()
            >> (LogisticRegression | KNeighborsClassifier)
        )
        run_hyperopt_on_planned_pipeline(plan)


class TestHyperopt(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_loguniform_search_space(self):
        from lale.search.schema2search_space import op_to_search_space
        from lale.search.search_space import SearchSpaceObject

        svc = SVC()
        space = op_to_search_space(svc)
        self.assertIsInstance(space, SearchSpaceObject)
        strobj = str(space)
        self.assertRegex(strobj, r"gamma->\[[^[]*,<log>,8]")
        pass

    def test_using_scoring(self):
        lr = LogisticRegression()
        clf = Hyperopt(estimator=lr, scoring="accuracy", cv=5, max_evals=1)
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_custom_scoring(self):
        from sklearn.metrics import f1_score

        lr = LogisticRegression()
        clf = Hyperopt(
            estimator=lr,
            scoring=make_scorer(f1_score, average="macro"),
            cv=5,
            max_evals=1,
        )
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_runtime_limit_hoc(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )
        X, y = load_iris(return_X_y=True)

        max_opt_time = 10.0
        hoc = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            scoring="accuracy",
            max_opt_time=max_opt_time,
        )
        start = time.time()
        _ = hoc.fit(X, y)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.7
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"

    def test_runtime_limit_zero_time_hoc(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )
        X, y = load_iris(return_X_y=True)

        hoc = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            scoring="accuracy",
            max_opt_time=0.0,
        )
        hoc_fitted = hoc.fit(X, y)
        assert hoc_fitted.get_pipeline() is None

    def test_runtime_limit_hor(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)

        max_opt_time = 3.0
        hor = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            max_opt_time=max_opt_time,
            scoring="r2",
        )
        start = time.time()
        _ = hor.fit(X[:500, :], y[:500])
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.2
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"

    def test_runtime_limit_zero_time_hor(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)

        hor = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            max_opt_time=0.0,
            scoring="r2",
        )
        hor_fitted = hor.fit(X, y)
        assert hor_fitted.get_pipeline() is None

    def test_hyperparam_overriding_with_hyperopt(self):
        pca1 = PCA(n_components=3)
        pca2 = PCA()
        search_space1 = hyperopt_search_space(pca1)
        search_space2 = hyperopt_search_space(pca2)
        self.assertNotEqual(search_space1, search_space2)

    def test_nested_pipeline1(self):
        data = load_iris()
        X, y = data.data, data.target
        # pipeline = KNeighborsClassifier() | (OneHotEncoder(handle_unknown = 'ignore') >> LogisticRegression())
        pipeline = KNeighborsClassifier() | (SimpleImputer() >> LogisticRegression())
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))

    def test_with_concat_features1(self):
        warnings.filterwarnings("ignore")

        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        pipeline = ((pca & nys) >> concat >> lr) | KNeighborsClassifier()
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))
        warnings.resetwarnings()

    def test_with_concat_features2(self):
        warnings.filterwarnings("ignore")

        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        from lale.operators import make_pipeline

        pipeline = make_pipeline(
            ((((SimpleImputer() | NoOp()) >> pca) & nys) >> concat >> lr)
            | KNeighborsClassifier()
        )
        clf = Hyperopt(estimator=pipeline, max_evals=1, handle_cv_failure=True)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))
        warnings.resetwarnings()

    def test_preprocessing_union(self):
        from lale.datasets import openml

        (train_X, train_y), (_test_X, _test_y) = openml.fetch(
            "credit-g", "classification", preprocess=False
        )

        prep_num = Project(columns={"type": "number"}) >> Normalizer
        if sklearn_version >= version.Version("1.2"):
            ohe = OneHotEncoder(sparse_output=False)
        else:
            ohe = OneHotEncoder(sparse=False)

        prep_cat = Project(columns={"not": {"type": "number"}}) >> ohe
        planned = (prep_num & prep_cat) >> ConcatFeatures >> RandomForestClassifier

        hyperopt_classifier = Hyperopt(estimator=planned, max_evals=1)
        _ = hyperopt_classifier.fit(train_X, train_y)

    def test_text_and_structured(self):
        from lale.datasets.uci.uci_datasets import fetch_drugslib

        train_X_all, train_y_all, _test_X, _test_y = fetch_drugslib()
        # subset to speed up debugging
        train_X, _train_X, train_y, _train_y = train_test_split(
            train_X_all, train_y_all, train_size=0.01, random_state=42
        )

        prep_text = Project(columns=["benefitsReview"]) >> TfidfVectorizer(
            max_features=100
        )
        prep_nums = Project(columns=["effectiveness"]) >> OrdinalEncoder()
        planned = (
            (prep_text & prep_nums)
            >> ConcatFeatures
            >> (LinearRegression | RandomForestRegressor)
        )

        hyperopt_classifier = Hyperopt(estimator=planned, max_evals=1, scoring="r2")
        _ = hyperopt_classifier.fit(train_X, train_y)

    def test_custom_scorer(self):
        pipeline = PCA() >> LogisticRegression()

        def custom_scorer(estimator, X, y, factor=0.1):
            # This is a custom scorer for demonstrating the use of kwargs
            # Just applies some factor to the accuracy
            predictions = estimator.predict(X)
            self.assertEqual(factor, 0.5)
            return factor * accuracy_score(y, predictions)

        clf = Hyperopt(
            estimator=pipeline,
            scoring=custom_scorer,
            cv=5,
            max_evals=1,
            args_to_scorer={"factor": 0.5},
        )
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_other_algorithms(self):
        for alg in ["rand", "tpe", "atpe", "anneal"]:
            hyperopt = Hyperopt(
                estimator=LogisticRegression, algo=alg, cv=3, max_evals=3
            )
            trained = hyperopt.fit(self.X_train, self.y_train)
            predictions = trained.predict(self.X_test)
            predictions_1 = hyperopt.predict(self.X_test)
            self.assertTrue(np.array_equal(predictions_1, predictions), alg)

    def test_args_to_cv(self):
        from sklearn.model_selection import GroupKFold

        data = pd.DataFrame(
            {
                "x_0": ["a"] * 10 + ["b"] * 10,
                "x_1": ["c"] * 18 + ["d"] * 2,
                "y": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            }
        )
        train_y = data.pop("y")
        train_x = data
        group_cols = train_x["x_0"]
        group_cols = [col_value[0] for col_value in group_cols.tolist()]
        group_kfold = GroupKFold(n_splits=2)
        pipeline = OneHotEncoder(handle_unknown="ignore") >> RandomForestClassifier()
        optimizer = Hyperopt(
            estimator=pipeline, cv=group_kfold, max_evals=1, verbose=True
        )
        trained_optimizer = optimizer.fit(
            train_x, train_y, args_to_cv={"groups": group_cols}
        )
        trained_optimizer.predict(train_x)

    def test_string_X(self):
        from sklearn.datasets import fetch_20newsgroups

        cats = ["alt.atheism", "sci.space"]
        newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)
        self.train_X, self.train_y = (
            np.array(newsgroups_train.data),
            newsgroups_train.target,
        )
        self.train_X = np.reshape(self.train_X, (self.train_X.shape[0], 1))
        newsgroups_test = fetch_20newsgroups(subset="test", categories=cats)
        self.test_X, self.test_y = (
            np.array(newsgroups_test.data),
            newsgroups_test.target,
        )
        self.test_X = np.reshape(self.test_X, (self.test_X.shape[0], 1))
        op = OpThatWorksWithFiles()
        optimizer = Hyperopt(estimator=op, cv=None, max_evals=2, verbose=True)
        with tempfile.NamedTemporaryFile(mode="w") as tmp:
            pd.DataFrame(self.train_X).to_csv(tmp.name, header=None, index=None)
            # np.savetxt(tmp.name, self.train_X, fmt="%s")
            with tempfile.NamedTemporaryFile(mode="w") as tmp1:
                pd.DataFrame(self.test_X).to_csv(tmp1.name, header=None, index=None)
                # np.savetxt(tmp1.name, self.test_X, fmt="%s")
                _ = optimizer.fit(
                    tmp.name,
                    y=self.train_y,
                    X_valid=tmp1.name,
                    y_valid=self.test_y,
                    sample_weight=[],
                )


class TestAutoConfigureClassification(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_Hyperopt(self):
        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=Hyperopt,
            cv=3,
            scoring="accuracy",
            max_evals=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert isinstance(best_pipeline, TrainedPipeline)

    def test_with_Hyperopt_2(self):
        choice = LogisticRegression | KNeighborsClassifier
        best = choice.auto_configure(
            self.X_train, self.y_train, optimizer=Hyperopt, cv=3, max_evals=3
        )
        _ = best.predict(self.X_test)

    def test_with_Hyperopt_3(self):
        planned_pipeline = (PCA() | Nystroem()) >> (
            LogisticRegression() | KNeighborsClassifier()
        )
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=Hyperopt,
            cv=3,
            scoring="accuracy",
            max_evals=10,
            frac_evals_with_defaults=0.2,
        )
        _ = best_pipeline.predict(self.X_test)
        assert isinstance(best_pipeline, TrainedPipeline)

    def test_with_gridsearchcv(self):
        warnings.simplefilter("ignore")
        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=GridSearchCV,
            cv=3,
            scoring="accuracy",
            lale_num_samples=1,
            lale_num_grids=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert best_pipeline is not None

    @unittest.skip(
        "SMAC interface is not currently supported with newer version of Python",
    )
    def test_with_smaccv(self):
        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=SMAC,
            cv=3,
            scoring="accuracy",
            max_evals=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert isinstance(best_pipeline, TrainedPipeline)


class TestAutoConfigureRegression(unittest.TestCase):
    def setUp(self):
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_Hyperopt(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=Hyperopt,
            cv=3,
            scoring="r2",
            max_evals=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert isinstance(best_pipeline, TrainedPipeline)

    def test_with_gridsearchcv(self):
        warnings.simplefilter("ignore")
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=GridSearchCV,
            cv=3,
            scoring="r2",
            lale_num_samples=1,
            lale_num_grids=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert best_pipeline is not None


class TestGridSearchCV(unittest.TestCase):
    def test_manual_grid(self):
        warnings.simplefilter("ignore")

        from lale import wrap_imported_operators

        wrap_imported_operators()
        iris = load_iris()
        parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
        svc = SVC()
        clf = GridSearchCV(estimator=svc, param_grid=parameters)
        clf.fit(iris.data, iris.target)
        clf.predict(iris.data)

    def test_with_gridsearchcv_auto_wrapped_pipe1(self):
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            clf = GridSearchCV(
                estimator=trainable,
                lale_num_samples=1,
                lale_num_grids=1,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_gridsearchcv_auto_wrapped_pipe2(self):
        lr = LogisticRegression()
        pca1 = PCA()
        pca1._name = "PCA1"
        pca2 = PCA()
        pca2._name = "PCA2"
        trainable = (pca1 | pca2) >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            clf = GridSearchCV(
                estimator=trainable,
                lale_num_samples=1,
                lale_num_grids=1,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_runtime_limit_hoc(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )

        X, y = load_iris(return_X_y=True)

        max_opt_time = 10.0
        hoc = GridSearchCV(
            estimator=planned_pipeline,
            cv=3,
            scoring="accuracy",
            max_opt_time=max_opt_time,
        )
        start = time.time()
        with self.assertRaises(BaseException):
            _ = hoc.fit(X, y)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.7
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"

    @unittest.skip("This test has finicky timing")
    def test_runtime_limit_hor(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)

        max_opt_time = 3
        hor = GridSearchCV(
            estimator=planned_pipeline,
            cv=3,
            max_opt_time=max_opt_time,
            scoring="r2",
        )
        start = time.time()
        with self.assertRaises(BaseException):
            _ = hor.fit(X[:500, :], y[:500])
        end = time.time()
        opt_time = end - start
        print(opt_time)
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.2
        ), f"Max time: {max_opt_time}, Actual time: {opt_time}, relative diff: {rel_diff}"


class TestCrossValidation(unittest.TestCase):
    def test_cv_folds(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()
        from sklearn.model_selection import KFold

        from lale.helpers import cross_val_score

        cv_results = cross_val_score(trainable_lr, iris.data, iris.target, cv=KFold(2))
        self.assertEqual(len(cv_results), 2)

    def test_cv_scoring(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()
        from sklearn.metrics import confusion_matrix

        from lale.helpers import cross_val_score

        cv_results = cross_val_score(
            trainable_lr, iris.data, iris.target, scoring=confusion_matrix
        )
        self.assertEqual(len(cv_results), 5)

    def test_cv_folds_scikit(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = load_iris()

        from sklearn.model_selection import KFold, cross_val_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_val_score(
                trainable_lr,
                iris.data,
                iris.target,
                cv=KFold(2),
                scoring=make_scorer(accuracy_score),
            )
        self.assertEqual(len(cv_results), 2)


class TestHigherOrderOperators(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_ada_boost(self):
        from lale.lib.sklearn import AdaBoostClassifier, DecisionTreeClassifier

        clf = AdaBoostClassifier(
            **with_fixed_estimator_name(estimator=DecisionTreeClassifier())
        )
        trained = clf.auto_configure(
            self.X_train, self.y_train, optimizer=Hyperopt, max_evals=1
        )

        est_name = get_sklearn_estimator_name()

        # Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(
            trained.hyperparams()[est_name].hyperparams()["min_samples_leaf"], 1
        )

    def test_ada_boost_pipe(self):
        from lale.lib.sklearn import AdaBoostClassifier, DecisionTreeClassifier

        clf = AdaBoostClassifier(
            **with_fixed_estimator_name(estimator=NoOp >> DecisionTreeClassifier())
        )
        trained = clf.auto_configure(
            self.X_train, self.y_train, optimizer=Hyperopt, max_evals=1
        )

        est_name = get_sklearn_estimator_name()

        # Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(
            trained.hyperparams()[est_name]
            .steps_list()[1]
            .hyperparams()["min_samples_leaf"],
            1,
        )

    def test_ada_boost1(self):
        from sklearn.tree import DecisionTreeClassifier

        from lale.lib.sklearn import AdaBoostClassifier

        clf = AdaBoostClassifier(
            **with_fixed_estimator_name(estimator=DecisionTreeClassifier())
        )
        clf.fit(self.X_train, self.y_train)

    def test_ada_boost_regressor(self):
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)
        X_train, _X_test, y_train, _y_test = train_test_split(X, y)
        from lale.lib.sklearn import AdaBoostRegressor, DecisionTreeRegressor

        reg = AdaBoostRegressor(
            **with_fixed_estimator_name(estimator=DecisionTreeRegressor())
        )
        trained = reg.auto_configure(
            X_train, y_train, optimizer=Hyperopt, max_evals=1, scoring="r2"
        )

        est_name = get_sklearn_estimator_name()

        # Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(
            trained.hyperparams()[est_name].hyperparams()["min_samples_leaf"], 1
        )

    def test_ada_boost_regressor_pipe(self):
        from lale.datasets.util import load_boston

        X, y = load_boston(return_X_y=True)
        X_train, _X_test, y_train, _y_test = train_test_split(X, y)
        from lale.lib.sklearn import AdaBoostRegressor, DecisionTreeRegressor

        reg = AdaBoostRegressor(
            **with_fixed_estimator_name(estimator=NoOp >> DecisionTreeRegressor())
        )
        trained = reg.auto_configure(
            X_train, y_train, optimizer=Hyperopt, max_evals=1, scoring="r2"
        )

        est_name = get_sklearn_estimator_name()

        # Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(
            trained.hyperparams()[est_name]
            .steps_list()[1]
            .hyperparams()["min_samples_leaf"],
            1,
        )


class TestSelectKBestTransformer(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_hyperopt(self):
        from lale.lib.sklearn import SelectKBest

        planned = SelectKBest >> LogisticRegression
        trained = planned.auto_configure(
            self.X_train,
            self.y_train,
            cv=3,
            optimizer=Hyperopt,
            max_evals=3,
            verbose=True,
        )
        _ = trained.predict(self.X_test)


class TestTopKVotingClassifier(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_fit_predict(self):
        ensemble = TopKVotingClassifier(
            estimator=(PCA() | Nystroem())
            >> (LogisticRegression() | KNeighborsClassifier()),
            args_to_optimizer={"max_evals": 3},
            k=2,
        )
        trained = ensemble.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_fit_args(self):
        ensemble = TopKVotingClassifier(
            estimator=(PCA() | Nystroem())
            >> (LogisticRegression() | KNeighborsClassifier()),
            k=2,
        )
        trained = ensemble.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_fit_smaller_trials(self):
        ensemble = TopKVotingClassifier(
            estimator=(PCA() | Nystroem())
            >> (LogisticRegression() | KNeighborsClassifier()),
            args_to_optimizer={"max_evals": 3},
            k=20,
        )
        trained = ensemble.fit(self.X_train, self.y_train)
        final_ensemble = trained._impl._best_estimator
        self.assertLessEqual(len(final_ensemble._impl_instance().estimators), 3)

    def test_fit_default_args(self):
        with self.assertRaises(ValueError):
            _ = TopKVotingClassifier()


class TestKNeighborsClassifier(unittest.TestCase):
    def setUp(self):
        all_X, all_y = load_iris(return_X_y=True)
        # 15 samples, small enough so folds are likely smaller than n_neighbors
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            all_X, all_y, train_size=15, test_size=None, shuffle=True, random_state=42
        )

    def test_schema_validation(self):
        trainable_16 = KNeighborsClassifier(n_neighbors=16)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = trainable_16.fit(self.train_X, self.train_y)
        trainable_15 = KNeighborsClassifier(n_neighbors=15)
        trained_15 = trainable_15.fit(self.train_X, self.train_y)
        _ = trained_15.predict(self.test_X)

    def test_hyperopt(self):
        planned = KNeighborsClassifier
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            cv=3,
            optimizer=Hyperopt,
            max_evals=3,
            verbose=True,
        )
        _ = trained.predict(self.test_X)

    def test_gridsearch(self):
        planned = KNeighborsClassifier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = planned.auto_configure(
                self.train_X, self.train_y, optimizer=GridSearchCV, cv=3
            )
        _ = trained.predict(self.test_X)

    @unittest.skip(
        "SMAC interface is not currently supported with newer version of Python",
    )
    def test_smac(self):
        planned = KNeighborsClassifier
        trained = planned.auto_configure(
            self.train_X, self.train_y, cv=3, optimizer=SMAC, max_evals=3
        )
        _ = trained.predict(self.test_X)


class TestKNeighborsRegressor(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_diabetes

        all_X, all_y = load_diabetes(return_X_y=True)
        # 15 samples, small enough so folds are likely smaller than n_neighbors
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            all_X, all_y, train_size=15, test_size=None, shuffle=True, random_state=42
        )

    def test_schema_validation(self):
        trainable_16 = KNeighborsRegressor(n_neighbors=16)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = trainable_16.fit(self.train_X, self.train_y)
        trainable_15 = KNeighborsRegressor(n_neighbors=15)
        trained_15 = trainable_15.fit(self.train_X, self.train_y)
        _ = trained_15.predict(self.test_X)

    def test_hyperopt(self):
        planned = KNeighborsRegressor
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            cv=3,
            optimizer=Hyperopt,
            max_evals=3,
            verbose=True,
            scoring="r2",
        )
        _ = trained.predict(self.test_X)

    def test_gridsearch(self):
        planned = KNeighborsRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = planned.auto_configure(
                self.train_X, self.train_y, optimizer=GridSearchCV, cv=3, scoring="r2"
            )
        _ = trained.predict(self.test_X)

    @unittest.skip(
        "SMAC interface is not currently supported with newer version of Python",
    )
    def test_smac(self):
        planned = KNeighborsRegressor
        trained = planned.auto_configure(
            self.train_X, self.train_y, cv=3, optimizer=SMAC, max_evals=3, scoring="r2"
        )
        _ = trained.predict(self.test_X)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        import scipy.sparse

        # from lale.datasets.data_schemas import add_schema
        all_X, all_y = load_iris(return_X_y=True)
        denseTrainX, self.test_X, self.train_y, self.test_y = train_test_split(
            all_X, all_y, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
        )
        # self.train_X = add_schema(scipy.sparse.csr_matrix(denseTrainX))
        self.train_X = scipy.sparse.csr_matrix(denseTrainX)

    def test_schema_validation(self):
        trainable_okay = StandardScaler(with_mean=False) >> LogisticRegression()
        _ = trainable_okay.fit(self.train_X, self.train_y)
        trainable_bad = StandardScaler(with_mean=True) >> LogisticRegression()
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = trainable_bad.fit(self.train_X, self.train_y)

    def test_hyperopt(self):
        planned = StandardScaler >> LogisticRegression().freeze_trainable()
        trained = planned.auto_configure(
            self.train_X,
            self.train_y,
            cv=3,
            optimizer=Hyperopt,
            max_evals=3,
            verbose=True,
            scoring="r2",
        )
        _ = trained.predict(self.test_X)

    def test_gridsearch(self):
        planned = StandardScaler >> LogisticRegression().freeze_trainable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = planned.auto_configure(
                self.train_X, self.train_y, optimizer=GridSearchCV, cv=3, scoring="r2"
            )
        _ = trained.predict(self.test_X)


class TestOptimizeLast(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_using_individual_operator(self):
        lr = LogisticRegression()  # Individual Operator
        trained_operator = lr.fit(self.X_train, self.y_train)

        # Now let's use Hyperopt to optimize the classifier
        hyperopt_args = {"scoring": "accuracy", "cv": 3, "max_evals": 2}
        opt_last = OptimizeLast(
            estimator=trained_operator,
            last_optimizer=Hyperopt,
            optimizer_args=hyperopt_args,
        )

        res_last = opt_last.fit(self.X_train, self.y_train)
        predictions = res_last.predict(self.X_test)
        predictions_1 = opt_last.predict(self.X_test)
        best_pipeline = res_last.get_pipeline()

        self.assertEqual(type(trained_operator), type(best_pipeline))
        assert np.array_equal(predictions_1, predictions)

    def test_using_pipeline(self):
        from sklearn.metrics import f1_score

        planned_pipeline = (PCA | NoOp) >> LogisticRegression

        # Let's first use Hyperopt to find the best pipeline
        opt = Hyperopt(estimator=planned_pipeline, max_evals=1)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        best_pipeline = res.get_pipeline()

        # Now let's use Hyperopt to optimize only the
        # last step (i.e., classifier) in the best pipeline
        hyperopt_args = {
            "scoring": make_scorer(f1_score, average="macro"),
            "cv": 3,
            "max_evals": 2,
        }
        opt_last = OptimizeLast(
            estimator=best_pipeline,
            last_optimizer=Hyperopt,
            optimizer_args=hyperopt_args,
        )

        res_last = opt_last.fit(self.X_train, self.y_train)
        predictions = res_last.predict(self.X_test)
        predictions_1 = opt_last.predict(self.X_test)
        best_pipeline2 = res_last.get_pipeline()

        self.assertEqual(type(best_pipeline), type(best_pipeline2))
        assert np.array_equal(predictions_1, predictions)

    def test_get_named_pipeline(self):
        pipeline = MinMaxScaler() >> KNeighborsClassifier()
        trained_pipeline = pipeline.fit(self.X_train, self.y_train)

        hyperopt_args = {"cv": 3, "max_evals": 2}
        opt_last = OptimizeLast(
            estimator=trained_pipeline,
            last_optimizer=Hyperopt,
            optimizer_args=hyperopt_args,
        )

        res_last = opt_last.fit(self.X_train, self.y_train)
        pipeline2 = res_last.get_pipeline(pipeline_name="p1")
        if pipeline2 is not None:
            trained_pipeline2 = pipeline2.fit(self.X_train, self.y_train)
            _ = trained_pipeline2.predict(self.X_test)

            self.assertEqual(type(trained_pipeline), type(trained_pipeline2))

    def test_unspecified_arguments(self):
        opt = OptimizeLast(optimizer_args={"max_evals": 1})  # No arguments
        res = opt.fit(self.X_train, self.y_train)
        predictions = res.predict(self.X_test)
        predictions_1 = opt.predict(self.X_test)
        best_pipeline = res.get_pipeline()

        assert np.array_equal(predictions_1, predictions)
        self.assertEqual(type(best_pipeline), TrainedIndividualOp)
