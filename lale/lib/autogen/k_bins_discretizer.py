import sklearn
from numpy import inf, nan
from packaging import version
from sklearn.preprocessing import KBinsDiscretizer as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator, sklearn_version


class _KBinsDiscretizerImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for KBinsDiscretizer    Bin continuous data into intervals.",
    "allOf": [
        {
            "type": "object",
            "required": ["n_bins", "encode", "strategy"],
            "relevantToOptimizer": ["encode", "strategy"],
            "additionalProperties": False,
            "properties": {
                "n_bins": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"type": "number"}},
                    ],
                    "default": 5,
                    "description": "The number of bins to produce",
                },
                "encode": {
                    "enum": ["onehot", "onehot-dense", "ordinal"],
                    "default": "onehot",
                    "description": "Method used to encode the transformed result",
                },
                "strategy": {
                    "enum": ["uniform", "quantile", "kmeans"],
                    "default": "quantile",
                    "description": "Strategy used to define the widths of the bins",
                },
            },
        },
        {
            "description": "A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array. ",
            "type": "object",
            "laleNot": "X/isSparse",
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fits the estimator.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "numeric array-like, shape (n_samples, n_features)",
            "description": "Data to be discretized.",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "ignored"},
    },
}

if sklearn_version >= version.Version("1.3"):
    _input_fit_schema["properties"]["sample_weight"] = {  # type:ignore
        "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
        "default": None,
        "description": 'Contains weight values to be associated with each sample. Only possible when strategy is set to "quantile".',
    }

_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Discretizes the data.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "laleType": "Any",
            "XXX TODO XXX": "numeric array-like, shape (n_samples, n_features)",
            "description": "Data to be discretized.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Data in the binned space.",
    "laleType": "Any",
    "XXX TODO XXX": "numeric array-like or sparse matrix",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.KBinsDiscretizer#sklearn-preprocessing-kbinsdiscretizer",
    "import_from": "sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
KBinsDiscretizer = make_operator(_KBinsDiscretizerImpl, _combined_schemas)

if sklearn_version >= version.Version("0.24"):
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.KBinsDiscretizer#sklearn-preprocessing-kbinsdiscretizer
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.preprocessing.KBinsDiscretizer#sklearn-preprocessing-kbinsdiscretizer
    KBinsDiscretizer = KBinsDiscretizer.customize_schema(
        dtype={
            "XXX TODO XXX": "dtype{np.float32, np.float64}, default=None",
            "laleType": "Any",
            "default": None,
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.3"):
    KBinsDiscretizer = KBinsDiscretizer.customize_schema(
        dtype={
            "XXX TODO XXX": "dtype{np.float32, np.float64}, default=None",
            "laleType": "Any",
            "default": None,
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.3"):
    KBinsDiscretizer = KBinsDiscretizer.customize_schema(
        subsample={
            "anyOf": [{"enum": ["warn", None]}, {"type": "integer", "minimum": 0}],
            "default": "warn",
            "description": "Maximum number of samples, used to fit the model, for computational efficiency. Defaults to 200_000 when strategy='quantile' and to None when strategy='uniform' or strategy='kmeans'. subsample=None means that all the training samples are used when computing the quantiles that determine the binning thresholds. Since quantile computation relies on sorting each column of X and that sorting has an n log(n) time complexity, it is recommended to use subsampling on datasets with a very large number of samples.",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.5"):
    KBinsDiscretizer = KBinsDiscretizer.customize_schema(
        subsample={
            "anyOf": [{"enum": [None]}, {"type": "integer", "minimum": 0}],
            "default": 20000,
            "description": "Maximum number of samples, used to fit the model, for computational efficiency. Defaults to 200_000 when strategy='quantile' and to None when strategy='uniform' or strategy='kmeans'. subsample=None means that all the training samples are used when computing the quantiles that determine the binning thresholds. Since quantile computation relies on sorting each column of X and that sorting has an n log(n) time complexity, it is recommended to use subsampling on datasets with a very large number of samples.",
        },
        set_as_available=True,
    )

set_docstrings(KBinsDiscretizer)
