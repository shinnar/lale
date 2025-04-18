from numpy import inf, nan
from packaging import version
from sklearn.decomposition import MiniBatchSparsePCA as Op
from sklearn.utils.metaestimators import available_if

from lale.docstrings import set_docstrings
from lale.operators import make_operator, sklearn_version


class _MiniBatchSparsePCAImpl:
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

    @available_if(lambda self: (hasattr(self._wrapped_model, "inverse_transform")))
    def inverse_transform(self, X):
        return self._wrapped_model.inverse_transform(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for MiniBatchSparsePCA    Mini-batch Sparse Principal Components Analysis",
    "allOf": [
        {
            "type": "object",
            "required": [
                "n_components",
                "alpha",
                "ridge_alpha",
                "n_iter",
                "callback",
                "batch_size",
                "verbose",
                "shuffle",
                "n_jobs",
                "method",
                "random_state",
            ],
            "relevantToOptimizer": [
                "n_components",
                "alpha",
                "n_iter",
                "batch_size",
                "shuffle",
                "method",
            ],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 256,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "number of sparse atoms to extract",
                },
                "alpha": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 2,
                    "distribution": "uniform",
                    "default": 1,
                    "description": "Sparsity controlling parameter",
                },
                "ridge_alpha": {
                    "type": "number",
                    "default": 0.01,
                    "description": "Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.",
                },
                "n_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "number of iterations to perform for each mini batch",
                },
                "callback": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "callable that gets invoked every five iterations",
                },
                "batch_size": {
                    "type": "integer",
                    "minimumForOptimizer": 3,
                    "maximumForOptimizer": 128,
                    "distribution": "uniform",
                    "default": 3,
                    "description": "the number of features to take in each mini batch",
                },
                "verbose": {
                    "anyOf": [{"type": "integer"}, {"type": "boolean"}],
                    "default": False,
                    "description": "Controls the verbosity; the higher, the more messages",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "whether to shuffle the data before splitting it in batches",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "Number of parallel jobs to run",
                },
                "method": {
                    "enum": ["lars", "cd"],
                    "default": "lars",
                    "description": "lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso)",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.",
                },
            },
        },
        {
            "description": "A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.",
            "type": "object",
            "laleNot": "X/isSparse",
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit the model from data in X.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vector, where n_samples in the number of samples and n_features is the number of features.",
        },
        "y": {},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Least Squares projection of the data onto the sparse components.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Test data to be transformed, must have the same number of features as the data used to train the model.",
        },
        "ridge_alpha": {
            "type": "number",
            "default": 0.01,
            "description": "Amount of ridge shrinkage to apply in order to improve conditioning",
        },
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data.",
    "laleType": "Any",
    "XXX TODO XXX": "X_new array, shape (n_samples, n_components)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.decomposition.MiniBatchSparsePCA#sklearn-decomposition-minibatchsparsepca",
    "import_from": "sklearn.decomposition",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
MiniBatchSparsePCA = make_operator(_MiniBatchSparsePCAImpl, _combined_schemas)

if sklearn_version >= version.Version("1.1"):
    MiniBatchSparsePCA = MiniBatchSparsePCA.customize_schema(
        max_no_improvement={
            "anyOf": [
                {
                    "type": "integer",
                    "minimum": 1,
                },
                {
                    "enum": [None],
                    "description": "Disable convergence detection based on cost function.",
                },
            ],
            "default": 10,
            "description": "Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed cost function.",
        },
        tol={
            "type": "number",
            "default": 0.001,
            "description": """Control early stopping based on the norm of the differences in the dictionary between 2 steps.

To disable early stopping based on changes in the dictionary, set tol to 0.0.""",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.2"):
    MiniBatchSparsePCA = MiniBatchSparsePCA.customize_schema(
        max_iter={
            "anyOf": [
                {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                },
                {"enum": [None]},
            ],
            "description": "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics. If max_iter is not None, n_iter is ignored.",
            "default": None,
        },
        n_iter={
            "anyOf": [
                {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                    "default": 1000,
                },
                {"enum": ["deprecated"]},
            ],
            "description": "total number of iterations to perform",
            "default": "deprecated",
        },
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.4"):
    MiniBatchSparsePCA = MiniBatchSparsePCA.customize_schema(
        max_iter={
            "anyOf": [
                {
                    "type": "integer",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 1000,
                    "distribution": "uniform",
                },
                {"enum": [None], "description": "deprecated"},
            ],
            "description": "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
            "default": 1000,
        },
        n_iter=None,
        set_as_available=True,
    )

if sklearn_version >= version.Version("1.6"):
    MiniBatchSparsePCA = MiniBatchSparsePCA.customize_schema(
        max_iter={
            "type": "integer",
            "minimumForOptimizer": 5,
            "maximumForOptimizer": 1000,
            "distribution": "uniform",
            "default": 1000,
            "description": "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
        },
        set_as_available=True,
    )

set_docstrings(MiniBatchSparsePCA)
