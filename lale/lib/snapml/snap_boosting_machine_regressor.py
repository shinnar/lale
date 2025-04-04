# Copyright 2019,2021 IBM Corporation
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
from packaging import version

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators

try:
    import snapml
    from snapml import SnapBoostingMachineRegressor as Base

    snapml_version = version.parse(getattr(snapml, "__version__"))
except ImportError:
    Base = None
    snapml_version = None


class _SnapBoostingMachineRegressorImpl:
    def __init__(self, **hyperparams):
        assert (
            snapml_version is not None and Base is not None
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""

        self._wrapped_model = Base(**hyperparams)

    def fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict(X, **predict_params)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "relevantToOptimizer": [
                "num_round",
                "learning_rate",
                "min_max_depth",
                "max_max_depth",
            ],
            "additionalProperties": False,
            "properties": {
                "num_round": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 100,
                    "maximumForOptimizer": 1000,
                    "default": 100,
                    "description": "Number of boosting iterations.",
                },
                "objective": {
                    "enum": ["mse", "cross_entropy"],
                    "default": "mse",
                    "description": "Training objective.",
                },
                "learning_rate": {
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "minimumForOptimizer": 0.01,
                    "maximumForOptimizer": 0.3,
                    "distribution": "uniform",
                    "default": 0.1,
                    "description": "Learning rate / shrinkage factor.",
                },
                "random_state": {
                    "type": "integer",
                    "default": 0,
                    "description": "Random seed.",
                },
                "colsample_bytree": {
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximum": 1.0,
                    "default": 1.0,
                    "description": "Fraction of feature columns used at each boosting iteration.",
                },
                "subsample": {
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximum": 1.0,
                    "default": 1.0,
                    "description": "Fraction of training examples used at each boosting iteration.",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Print off information during training.",
                },
                "lambda_l2": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.0,
                    "description": "L2-reguralization penalty used during tree-building.",
                },
                "early_stopping_rounds": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "description": "When a validation set is provided, training will stop if the validation loss does not increase after a fixed number of rounds.",
                },
                "compress_trees": {
                    "type": "boolean",
                    "default": False,
                    "description": "Compress trees after training for fast inference.",
                },
                "base_score": {
                    "anyOf": [
                        {
                            "type": "number",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Base score to initialize boosting algorithm. If None then the algorithm will initialize the base score to be the the logit of the probability of the positive class.",
                },
                "max_depth": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If set, will set min_max_depth = max_depth = max_max_depth",
                },
                "min_max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 5,
                    "default": 1,
                    "description": "Minimum max_depth of trees in the ensemble.",
                },
                "max_max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 10,
                    "default": 5,
                    "description": "Maximum max_depth of trees in the ensemble.",
                },
                "n_jobs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Number of threads to use during training.",
                },
                "use_histograms": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use histograms to accelerate tree-building.",
                },
                "hist_nbins": {
                    "type": "integer",
                    "default": 256,
                    "description": "Number of histogram bins.",
                },
                "use_gpu": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use GPU for tree-building.",
                },
                "gpu_id": {
                    "type": "integer",
                    "default": 0,
                    "description": "Device ID for GPU to use during training.",
                },
                "tree_select_probability": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0,
                    "description": "Probability of selecting a tree (rather than a kernel ridge regressor) at each boosting iteration.",
                },
                "regularizer": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "description": "L2-regularization penality for the kernel ridge regressor.",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include intercept term in the kernel ridge regressor.",
                },
                "gamma": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "description": "Guassian kernel parameter.",
                },
                "n_components": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "description": "Number of components in the random projection.",
                },
            },
        },
        {
            "description": "GPU only supported for histogram-based splits.",
            "anyOf": [
                {"type": "object", "properties": {"use_gpu": {"enum": [False]}}},
                {"type": "object", "properties": {"use_histograms": {"enum": [True]}}},
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Build a boosted ensemble from the training set (X, y).",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "y": {
            "description": "The regression target.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
            ],
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
            "default": None,
        },
        "X_val": {
            "anyOf": [
                {
                    "type": "array",
                    "description": "The outer array is over validation samples aka rows.",
                    "items": {
                        "type": "array",
                        "description": "The inner array is over features aka columns.",
                        "items": {"type": "number"},
                    },
                },
                {"enum": [None], "description": "No validation set provided."},
            ],
            "default": None,
        },
        "y_val": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "No validation set provided."},
            ],
            "description": "The validation regression target.",
            "default": None,
        },
        "sample_weight_val": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {
                    "enum": [None],
                    "description": "Validation samples are equally weighted.",
                },
            ],
            "description": "Validation sample weights.",
            "default": None,
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "n_jobs": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "Number of threads used to run inference.",
        },
    },
}

_output_predict_schema = {
    "description": "The predicted values.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Boosting machine Regressor`_ from `Snap ML`_.

.. _`Boosting machine Regressor`: https://snapml.readthedocs.io/en/latest/#snapml.BoostingMachineRegressor
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.snap_boosting_machine_regressor.html",
    "import_from": "snapml",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


SnapBoostingMachineRegressor = lale.operators.make_operator(
    _SnapBoostingMachineRegressorImpl, _combined_schemas
)

if snapml_version is not None and snapml_version >= version.Version("1.12"):
    SnapBoostingMachineRegressor = SnapBoostingMachineRegressor.customize_schema(
        max_delta_step={
            "description": """Regularization term to ensure numerical stability.""",
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0.0,
                },
                {"enum": [None]},
            ],
            "default": 0.0,
        }
    )

if snapml_version is not None and snapml_version >= version.Version("1.14"):
    SnapBoostingMachineRegressor = SnapBoostingMachineRegressor.customize_schema(
        alpha={
            "type": "number",
            "minimum": 0.0,
            "exclusiveMinimum": True,
            "minimumForOptimizer": 1e-10,
            "maximumForOptimizer": 1.0,
            "default": 0.5,
            "description": "Quantile used when 'objective = quantile'.",
        },
        min_h_quantile={
            "type": "number",
            "minimumForOptimizer": 0.0,
            "maximumForOptimizer": 1.0,
            "default": 0.0,
            "description": "Regularization term for quantile regression.",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(SnapBoostingMachineRegressor)
