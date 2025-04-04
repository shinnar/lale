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

import pandas as pd
from packaging import version
from sklearn.ensemble import AdaBoostClassifier as SKLModel

import lale.docstrings
import lale.operators
from lale.helpers import get_estimator_param_name_from_hyperparams

from .fit_spec_proxy import _FitSpecProxy
from .function_transformer import FunctionTransformer


class _AdaBoostClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        est_name = get_estimator_param_name_from_hyperparams(self._hyperparams)

        base_estimator = hyperparams.get(est_name, None)
        if base_estimator is None:
            estimator_impl = None
        else:
            estimator_impl = _FitSpecProxy(base_estimator)

        base_hyperparams = {est_name: estimator_impl}

        self._wrapped_model = SKLModel(**{**hyperparams, **base_hyperparams})

    def get_params(self, deep=True):
        out = self._wrapped_model.get_params(deep=deep)
        # we want to return the lale operator, not the underlying impl
        est_name = get_estimator_param_name_from_hyperparams(self._hyperparams)
        out[est_name] = self._hyperparams[est_name]
        return out

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            feature_transformer = FunctionTransformer(
                func=lambda X_prime: pd.DataFrame(X_prime, columns=X.columns),
                inverse_func=None,
                check_inverse=False,
            )

            est_name = get_estimator_param_name_from_hyperparams(self._hyperparams)
            self._hyperparams[est_name] = _FitSpecProxy(
                feature_transformer >> self._hyperparams[est_name]
            )
            self._wrapped_model = SKLModel(**self._hyperparams)
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def predict_log_proba(self, X):
        return self._wrapped_model.predict_log_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)

    def score(self, X, y, sample_weight=None):
        return self._wrapped_model.score(X, y, sample_weight)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "base_estimator",
                "n_estimators",
                "learning_rate",
                "algorithm",
                "random_state",
            ],
            "relevantToOptimizer": ["n_estimators", "learning_rate", "algorithm"],
            "additionalProperties": False,
            "properties": {
                "base_estimator": {
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                    "description": "The base estimator from which the boosted ensemble is built.",
                },
                "n_estimators": {
                    "type": "integer",
                    "minimumForOptimizer": 50,
                    "maximumForOptimizer": 500,
                    "distribution": "uniform",
                    "default": 50,
                    "description": "The maximum number of estimators at which boosting is terminated.",
                },
                "learning_rate": {
                    "type": "number",
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 1.0,
                    "distribution": "loguniform",
                    "default": 1.0,
                    "description": "Learning rate shrinks the contribution of each classifier by",
                },
                "algorithm": {
                    "enum": ["SAMME", "SAMME.R"],
                    "default": "SAMME.R",
                    "description": "If 'SAMME.R' then use the SAMME.R real boosting algorithm.",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator;",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Build a boosted classifier from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "The target values (class labels).",
        },
        "sample_weight": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Sample weights. If None, the sample weights are initialized to",
        },
    },
}
_input_predict_schema = {
    "description": "Predict classes for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
    },
}
_output_predict_schema = {
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Predict class probabilities for X.",
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "The training input samples. Sparse matrix can be CSC, CSR, COO,",
        },
    },
}
_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples. The order of",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "number"},
    },
}

_input_decision_function_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_decision_function_schema = {
    "description": "Confidence scores for samples for each class in the model.",
    "anyOf": [
        {
            "description": "In the multi-way case, score per (sample, class) combination.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        {
            "description": "In the binary case, score for `self._classes[1]`.",
            "type": "array",
            "items": {"type": "number"},
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`AdaBoost classifier`_ from scikit-learn for boosting ensemble.

.. _`AdaBoost classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.ada_boost_classifier.html",
    "import_from": "sklearn.ensemble",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

AdaBoostClassifier = lale.operators.make_operator(
    _AdaBoostClassifierImpl, _combined_schemas
)


if lale.operators.sklearn_version >= version.Version("1.2"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        base_estimator={
            "anyOf": [
                {"laleType": "operator"},
                {"enum": ["deprecated"]},
            ],
            "default": "deprecated",
            "description": "Deprecated. Use `estimator` instead.",
        },
        estimator={
            "anyOf": [
                {"laleType": "operator"},
                {"enum": [None], "description": "DecisionTreeClassifier"},
            ],
            "default": None,
            "description": "The base estimator to fit on random subsets of the dataset.",
        },
        constraint={
            "description": "Only `estimator` or `base_estimator` should be specified.  As `base_estimator` is deprecated, use `estimator`.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"base_estimator": {"enum": [False, "deprecated"]}},
                },
                {
                    "type": "object",
                    "properties": {
                        "estimator": {"enum": [None]},
                    },
                },
            ],
        },
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.3"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        base_estimator={
            "anyOf": [
                {"laleType": "operator"},
                {"enum": ["deprecated", None]},
            ],
            "default": "deprecated",
            "description": "Deprecated. Use `estimator` instead.",
        },
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.4"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        base_estimator=None,
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.4"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        algorithm={
            "anyOf": [
                {
                    "enum": ["SAMME"],
                    "description": "Use the SAMME discrete boosting algorithm.",
                },
                {"enum": ["SAMME.R"], "description": "deprecated"},
            ],
            "default": "SAMME.R",
            "description": "The boosting algorithm to use",
        },
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.6"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        algorithm={
            "anyOf": [
                {
                    "enum": ["SAMME"],
                },
                {"enum": ["deprecated"]},
            ],
            "default": "deprecated",
            "description": "deprecated",
        },
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("1.8"):
    AdaBoostClassifier = AdaBoostClassifier.customize_schema(
        algorithm=None,
        set_as_available=True,
    )


lale.docstrings.set_docstrings(AdaBoostClassifier)
