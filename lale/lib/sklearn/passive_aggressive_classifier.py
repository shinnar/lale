# Copyright 2019-2022 IBM Corporation
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

import sklearn
import sklearn.linear_model
from packaging import version

import lale.docstrings
import lale.operators
from lale.schemas import Int

from ._common_schemas import schema_1D_cats, schema_2D_numbers, schema_X_numbers

_hyperparams_schema = {
    "description": "Passive Aggressive Classifier",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "C",
                "fit_intercept",
                "max_iter",
                "tol",
                "early_stopping",
                "shuffle",
                "loss",
                "average",
            ],
            "relevantToOptimizer": [
                "C",
                "fit_intercept",
                "max_iter",
                "tol",
                "early_stopping",
                "shuffle",
                "loss",
                "average",
            ],
            "properties": {
                "C": {
                    "type": "number",
                    "description": "Maximum step size (regularization). Defaults to 1.0.",
                    "default": 1.0,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-5,
                    "maximumForOptimizer": 10,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the intercept should be estimated or not. If False, the"
                    "the data is assumed to be already centered.",
                },
                "max_iter": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 5,
                            "maximumForOptimizer": 1000,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": 5,
                    "description": "The maximum number of passes over the training data (aka epochs).",
                },
                "tol": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 1e-08,
                            "maximumForOptimizer": 0.01,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,  # default value is 1e-3 from sklearn 0.21.
                    "description": "The stopping criterion. If it is not None, the iterations will stop",
                },
                "early_stopping": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use early stopping to terminate training when validation.",
                },
                "validation_fraction": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.1,
                    "description": "The proportion of training data to set aside as validation set for early stopping.",
                },
                "n_iter_no_change": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 10,
                    "default": 5,
                    "description": "Number of iterations with no improvement to wait before early stopping.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether or not the training data should be shuffled after each epoch.",
                },
                "verbose": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 0,
                    "description": "The verbosity level",
                },
                "loss": {
                    "enum": ["hinge", "squared_hinge"],
                    "default": "hinge",
                    "description": "The loss function to be used:",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "The number of CPUs to use to do the OVA (One Versus All, for",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The seed of the pseudo random number generator to use when shuffling",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "When set to True, reuse the solution of the previous call to"
                    " fit as initialization, otherwise, just erase the previous solution.",
                },
                "class_weight": {
                    "anyOf": [{"type": "object"}, {"enum": ["balanced", None]}],
                    "default": None,
                    "description": "Preset for the class_weight fit parameter.",
                },
                "average": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer", "forOptimizer": False},
                    ],
                    "default": False,
                    "description": "When set to True, computes the averaged SGD weights and stores the result in the ``coef_`` attribute.",
                },
                "n_iter": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximumForOptimizer": 10},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The number of passes over the training data (aka epochs).",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "description": "Fit linear model with Passive Aggressive algorithm.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "coef_init": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "The initial coefficients to warm-start the optimization.",
        },
        "intercept_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "The initial intercept to warm-start the optimization.",
        },
    },
}

_input_partial_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "classes": schema_1D_cats,
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
    "description": """`Passive aggressive`_ classifier from scikit-learn.

.. _`Passive aggressive`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.passive_aggressive_classifier.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_partial_fit": _input_partial_fit_schema,
        "input_predict": schema_X_numbers,
        "output_predict": schema_1D_cats,
        "input_decision_function": schema_X_numbers,
        "output_decision_function": _output_decision_function_schema,
    },
}

PassiveAggressiveClassifier: lale.operators.PlannedIndividualOp
PassiveAggressiveClassifier = lale.operators.make_operator(
    sklearn.linear_model.PassiveAggressiveClassifier, _combined_schemas
)


if lale.operators.sklearn_version >= version.Version("0.21"):
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
    PassiveAggressiveClassifier = PassiveAggressiveClassifier.customize_schema(
        max_iter=Int(
            minimumForOptimizer=5,
            maximumForOptimizer=1000,
            distribution="uniform",
            desc="The maximum number of passes over the training data (aka epochs).",
            default=1000,
        ),
        set_as_available=True,
    )

if lale.operators.sklearn_version >= version.Version("0.22"):
    # old: https://scikit-learn.org/0.21/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
    # new: https://scikit-learn.org/0.22/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
    PassiveAggressiveClassifier = PassiveAggressiveClassifier.customize_schema(
        n_iter=None, set_as_available=True
    )

if lale.operators.sklearn_version >= version.Version("1.7"):
    PassiveAggressiveClassifier = PassiveAggressiveClassifier.customize_schema(
        average={
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "integer",
                    "forOptimizer": False,
                    "minimum": 0,
                    "exclusiveMinimum": True,
                },
            ],
            "default": False,
            "description": "When set to True, computes the averaged SGD weights and stores the result in the ``coef_`` attribute.",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(PassiveAggressiveClassifier)
