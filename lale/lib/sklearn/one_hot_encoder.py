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
import scipy.sparse
import sklearn
import sklearn.preprocessing
from packaging import version

import lale.docstrings
import lale.operators
from lale.schemas import AnyOf, Bool, Enum

sklearn_version = version.parse(getattr(sklearn, "__version__"))

_hyperparams_schema = {
    "description": "Hyperparameter schema for the OneHotEncoder model from scikit-learn.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["categories", "dtype", "handle_unknown"],
            "relevantToOptimizer": [],
            "properties": {
                "categories": {
                    "anyOf": [
                        {
                            "description": "Determine categories automatically from training data.",
                            "enum": ["auto", None],
                        },
                        {
                            "description": "The ith list element holds the categories expected in the ith column.",
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "Should be sorted.",
                                    },
                                ]
                            },
                        },
                    ],
                    "default": "auto",
                },
                "sparse": {
                    "description": "Will return sparse matrix if set true, else array.",
                    "type": "boolean",
                    "default": True,
                },
                "dtype": {
                    "description": "Desired dtype of output, must be number. See https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html#arrays-scalars-built-in",
                    "laleType": "Any",
                    "default": "float64",
                },
                "handle_unknown": {
                    "description": "Whether to raise an error or ignore if an unknown categorical feature is present during transform.",
                    "enum": ["error", "ignore"],
                    "default": "error",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"description": "Target class labels; the array is over samples."},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {
    "description": "One-hot codes.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`One-hot encoder`_ transformer from scikit-learn that encodes categorical features as numbers.

.. _`One-hot encoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.one_hot_encoder.html",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


class _OneHotEncoderImpl:
    def __init__(self, **hyperparams):
        self._wrapped_model = sklearn.preprocessing.OneHotEncoder(**hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        if isinstance(X, pd.DataFrame):
            self._X_columns = X.columns
        return self

    def transform(self, X):
        result = self._wrapped_model.transform(X)
        if isinstance(X, pd.DataFrame):
            if sklearn_version >= version.Version("1.0"):
                columns = self._wrapped_model.get_feature_names_out(X.columns)
            else:
                columns = self._wrapped_model.get_feature_names(X.columns)
            if isinstance(result, scipy.sparse.csr_matrix):
                result = result.toarray()
            result = pd.DataFrame(data=result, index=X.index, columns=columns)
        return result

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        is_fitted = hasattr(self._wrapped_model, "categories_")
        if not is_fitted:
            return _output_transform_schema
        in_names = None
        if "items" in s_X and "items" in s_X["items"]:
            col_schemas = s_X["items"]["items"]
            if isinstance(col_schemas, list):
                desc = [s.get("description", "") for s in col_schemas]
                if "" not in desc and len(desc) == len(set(desc)):
                    in_names = desc
        if in_names is None and hasattr(self, "_X_columns"):
            in_names = self._X_columns
        if in_names is None:
            return _output_transform_schema
        if sklearn_version >= version.Version("1.0"):
            out_names = self._wrapped_model.get_feature_names_out(in_names)
        else:
            out_names = self._wrapped_model.get_feature_names(in_names)
        result = {
            **s_X,
            "items": {
                **(s_X.get("items", {})),
                "minItems": len(out_names),
                "maxItems": len(out_names),
                "items": [{"description": n, "type": "number"} for n in out_names],
            },
        }
        return result


OneHotEncoder = lale.operators.make_operator(_OneHotEncoderImpl, _combined_schemas)

if sklearn_version >= version.Version("0.21"):
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        drop={
            "anyOf": [
                {"enum": ["first"]},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "forOptimizer": False,
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Specifies a methodology to use to drop one of the categories per feature.",
        },
        set_as_available=True,
    )
if version.Version("0.21") <= sklearn_version < version.Version("1.0"):
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        constraint={
            "description": "'handle_unknown' must be 'error' when the drop parameter is specified, as both would create categories that are all zero.",
            "anyOf": [
                {"type": "object", "properties": {"drop": {"enum": [None]}}},
                {
                    "type": "object",
                    "properties": {"handle_unknown": {"enum": ["error"]}},
                },
            ],
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("0.23"):
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        drop={
            "anyOf": [
                {"enum": ["first", "if_binary"]},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "forOptimizer": False,
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "Specifies a methodology to use to drop one of the categories per feature.",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.1"):
    # new: https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        handle_unknown={
            "description": "Specifies the way unknown categories are handled during transform.",
            "anyOf": [
                {
                    "enum": ["error"],
                    "description": "Raise an error if an unknown category is present during transform.",
                },
                {
                    "enum": ["ignore"],
                    "description": "When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros. In the inverse transform, an unknown category will be denoted as None.",
                },
                {
                    "enum": ["infrequent_if_exist"],
                    "description": "When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will map to the infrequent category if it exists. The infrequent category will be mapped to the last position in the encoding. During inverse transform, an unknown category will be mapped to the category denoted ``'infrequent'`` if it exists. If the ``'infrequent'`` category does not exist, then transform and inverse_transform will handle an unknown category as with ``handle_unknown='ignore'``. Infrequent categories exist based on ``min_frequency`` and ``max_categories``. Read more in the User Guide.",
                },
            ],
            "default": "error",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.2"):
    # new: https://scikit-learn.org/1.2/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        sparse=AnyOf(
            [
                Bool(
                    desc="Will return sparse matrix if set true, else array.",
                    default=True,
                    forOptimizer=False,
                ),
                Enum(values=["deprecated"]),
            ],
            default="deprecated",
        ),
        sparse_output=Bool(
            desc="Will return sparse matrix if set true, else will return an array.",
            default=True,
        ),
    )

if sklearn_version >= version.Version("1.3"):
    # new: https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(
        feature_name_combiner={
            "anyOf": [
                {
                    "enum": ["concat"],
                    "description": 'concatenates encoded feature name and category with feature + "_" + str(category).E.g. feature X with values 1, 6, 7 create feature names X_1, X_6, X_7.',
                },
                {
                    "laleType": "callable",
                    "forOptimizer": False,
                    "description": "Callable with signature def callable(input_feature, category) that returns a string",
                },
            ],
            "default": "concat",
            "description": "Used to create feature names to be returned by get_feature_names_out.",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.4"):
    # new: https://scikit-learn.org/1.2/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    OneHotEncoder = OneHotEncoder.customize_schema(sparse=None, set_as_available=True)

if sklearn_version >= version.Version("1.6"):
    OneHotEncoder = OneHotEncoder.customize_schema(
        handle_unknown={
            "description": "Specifies the way unknown categories are handled during transform.",
            "anyOf": [
                {
                    "enum": ["error"],
                    "description": "Raise an error if an unknown category is present during transform.",
                },
                {
                    "enum": ["warn"],
                    "description": 'When an unknown category is encountered during transform a warning is issued, and the encoding then proceeds as described for handle_unknown="infrequent_if_exist".',
                },
                {
                    "enum": ["ignore"],
                    "description": "When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros. In the inverse transform, an unknown category will be denoted as None.",
                },
                {
                    "enum": ["infrequent_if_exist"],
                    "description": "When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will map to the infrequent category if it exists. The infrequent category will be mapped to the last position in the encoding. During inverse transform, an unknown category will be mapped to the category denoted ``'infrequent'`` if it exists. If the ``'infrequent'`` category does not exist, then transform and inverse_transform will handle an unknown category as with ``handle_unknown='ignore'``. Infrequent categories exist based on ``min_frequency`` and ``max_categories``. Read more in the User Guide.",
                },
            ],
            "default": "error",
        },
        set_as_available=True,
    )

lale.docstrings.set_docstrings(OneHotEncoder)
