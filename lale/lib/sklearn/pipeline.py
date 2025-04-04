# Copyright 2020 IBM Corporation
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

import logging
import typing

from packaging import version
from sklearn.utils.metaestimators import available_if

import lale.docstrings
import lale.helpers
import lale.operators
from lale.schemas import Bool

logger = logging.getLogger(__name__)


def _pipeline_has(attr):
    return lambda self: (hasattr(self._pipeline, attr))


class _PipelineImpl:
    def __init__(self, **hyperparams):
        if hyperparams.get("memory", None):
            logger.warning("Caching is not yet implemented.")
        if hyperparams.get("verbose", False):
            logger.warning(
                "Verbose is not implemented; instead, use lale.operators.logger.setLevel(logging.INFO)."
            )
        self._names = [name for name, _ in hyperparams["steps"]]
        new_steps = []
        for _, op in hyperparams["steps"]:
            if op is None or op == "passthrough":
                from lale.lib.lale import NoOp

                new_steps.append(NoOp)
            else:
                new_steps.append(op)
        self._pipeline = lale.operators.make_pipeline(*new_steps)
        self._final_estimator = self._pipeline.get_last()

    def fit(self, X, y=None, **fit_params):
        if y is None:
            self._pipeline = self._pipeline.fit(X, **fit_params)
        else:
            self._pipeline = self._pipeline.fit(X, y, **fit_params)
        self._final_estimator = self._pipeline.get_last()
        return self

    @available_if(_pipeline_has("predict"))
    def predict(self, X, **predict_params):
        result = self._pipeline.predict(X, **predict_params)
        return result

    @available_if(_pipeline_has("predict_proba"))
    def predict_proba(self, X):
        result = self._pipeline.predict_proba(X)
        return result

    @available_if(_pipeline_has("transform"))
    def transform(self, X, y=None):
        if y is None:
            result = self._pipeline.transform(X)
        else:
            result = self._pipeline.transform(X, y)
        return result

    def viz_label(self) -> str:
        return "Pipeline: " + ", ".join(self._names)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["steps"],
            "relevantToOptimizer": [],
            "properties": {
                "steps": {
                    "description": "List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an estimator.",
                    "type": "array",
                    "items": {
                        "description": "Tuple of (name, transform).",
                        "type": "array",
                        "laleType": "tuple",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": [
                            {"description": "Name.", "type": "string"},
                            {
                                "anyOf": [
                                    {
                                        "description": "Transform.",
                                        "laleType": "operator",
                                    },
                                    {
                                        "description": "NoOp",
                                        "enum": [None, "passthrough"],
                                    },
                                ]
                            },
                        ],
                    },
                },
                "memory": {
                    "description": "Used to cache the fitted transformers of the pipeline.",
                    "anyOf": [
                        {
                            "description": "Path to the caching directory.",
                            "type": "string",
                        },
                        {
                            "description": "Object with the joblib.Memory interface",
                            "type": "object",
                            "forOptimizer": False,
                        },
                        {"description": "No caching.", "enum": [None]},
                    ],
                    "default": None,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {"description": "Features.", "laleType": "Any"},
        "y": {"description": "Target for supervised learning.", "laleType": "Any"},
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Features.", "laleType": "Any"}},
}

_output_predict_schema = {
    "description": "Predictions.",
    "laleType": "Any",
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Features.", "laleType": "Any"}},
}

_output_predict_proba_schema = {
    "description": "Probability of the sample for each class in the model.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {"description": "Features.", "laleType": "Any"}},
}

_output_transform_schema = {
    "description": "Features.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Pipeline_ from scikit-learn creates a sequential list of operators.

.. _Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pipeline.html",
    "import_from": "sklearn.pipeline",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Pipeline = lale.operators.make_operator(_PipelineImpl, _combined_schemas)

if lale.operators.sklearn_version >= version.Version("0.21"):
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.pipeline.Pipeline.html
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.pipeline.Pipeline.html
    Pipeline = typing.cast(
        lale.operators.PlannedIndividualOp,
        Pipeline.customize_schema(
            verbose=Bool(
                desc="If True, the time elapsed while fitting each step will be printed as it is completed.",
                default=False,
            ),
            set_as_available=True,
        ),
    )

if lale.operators.sklearn_version >= version.Version("1.6"):
    Pipeline = typing.cast(
        lale.operators.PlannedIndividualOp,
        Pipeline.customize_schema(
            transform_input={
                "anyOf": [
                    {"enum": [None]},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "forOptimizer": False,
                    },
                ],
                "description": """
The names of the metadata parameters that should be transformed by the pipeline before passing it to the step consuming it.
This enables transforming some input arguments to fit (other than X) to be transformed by the steps of the pipeline up to the step which requires them. Requirement is defined via metadata routing. For instance, this can be used to pass a validation set through the pipeline.
You can only set this if metadata routing is enabled, which you can enable using sklearn.set_config(enable_metadata_routing=True).""",
                "default": None,
            },
            set_as_available=True,
        ),
    )

lale.docstrings.set_docstrings(Pipeline)
