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
    from snapml import SnapBoostingMachineClassifier as Base

    snapml_version = version.parse(getattr(snapml, "__version__"))

except ImportError:
    Base = None
    snapml_version = None


class _SnapBoostingMachineClassifierImpl:
    def __init__(self, **hyperparams):
        assert (
            snapml_version is not None and Base is not None
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""

        if (
            snapml_version > version.Version("1.7.8")
            and hyperparams.get("gpu_ids", None) is None
        ):
            hyperparams["gpu_ids"] = [0]

        self._wrapped_model = Base(**hyperparams)

    def fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict_proba(X, **predict_proba_params)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "required": [
                "num_round",
                "learning_rate",
                "random_state",
                "colsample_bytree",
                "subsample",
                "verbose",
                "lambda_l2",
                "early_stopping_rounds",
                "compress_trees",
                "base_score",
                "class_weight",
                "max_depth",
                "min_max_depth",
                "max_max_depth",
                "n_jobs",
                "use_histograms",
                "hist_nbins",
                "use_gpu",
                "gpu_id",
                "tree_select_probability",
                "regularizer",
                "fit_intercept",
                "gamma",
                "n_components",
            ],
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
                "class_weight": {
                    "enum": ["balanced", None],
                    "default": None,
                    "description": "If set to 'balanced' samples weights will be applied to account for class imbalance, otherwise no sample weights will be used.",
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
            "description": "The classes.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
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
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
                {"enum": [None], "description": "No validation set provided."},
            ],
            "description": "The validation classes.",
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
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
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

_output_predict_proba_schema = {
    "type": "array",
    "description": "The outer array is over samples aka rows.",
    "items": {
        "type": "array",
        "description": "The inner array contains probabilities corresponding to each class.",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Boosting machine classifier`_ from `Snap ML`_. It can be used for binary classification problems.

.. _`Boosting machine classifier`: https://snapml.readthedocs.io/en/latest/#snapml.BoostingMachineClassifier
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.snap_boosting_machine_classifier.html",
    "import_from": "snapml",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}


SnapBoostingMachineClassifier = lale.operators.make_operator(
    _SnapBoostingMachineClassifierImpl, _combined_schemas
)

if snapml_version is not None and snapml_version > version.Version("1.7.8"):  # type: ignore # noqa
    SnapBoostingMachineClassifier = SnapBoostingMachineClassifier.customize_schema(
        gpu_id=None,
        gpu_ids={
            "description": "Device IDs of the GPUs which will be used when GPU acceleration is enabled.",
            "anyOf": [
                {"type": "array", "items": {"type": "integer"}},
                {"enum": [None], "description": "Use [0]."},
            ],
            "default": None,
            "forOptimizer": False,
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(SnapBoostingMachineClassifier)
