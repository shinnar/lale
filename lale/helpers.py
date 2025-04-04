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

import ast
import copy
import importlib
import logging
import time
import traceback
from importlib import util
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.pipeline
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, check_scoring, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.metaestimators import _safe_split

import lale.datasets.data_schemas

try:
    import torch
except ImportError:
    torch = None

spark_loader = util.find_spec("pyspark")
spark_installed = spark_loader is not None
if spark_installed:
    from pyspark.sql.dataframe import DataFrame as spark_df
else:
    spark_df = None

logger = logging.getLogger(__name__)

LALE_NESTED_SPACE_KEY = "__lale_nested_space"

astype_type = Literal["lale", "sklearn"]
datatype_param_type = Literal["pandas", "spark"]
randomstate_type = Union[RandomState, int, None]


def make_nested_hyperopt_space(sub_space):
    return {LALE_NESTED_SPACE_KEY: sub_space}


def assignee_name(level=1) -> Optional[str]:
    tb = traceback.extract_stack()
    file_name, _line_number, _function_name, text = tb[-(level + 2)]
    try:
        tree = ast.parse(text, file_name)
    except SyntaxError:
        return None
    assert tree is not None and isinstance(tree, ast.Module)
    if len(tree.body) == 1:
        stmt = tree.body[0]
        if isinstance(stmt, ast.Assign):
            lhs = stmt.targets
            if len(lhs) == 1:
                res = lhs[0]
                if isinstance(res, ast.Name):
                    return res.id
    return None


def arg_name(pos=0, level=1) -> Optional[str]:
    tb = traceback.extract_stack()
    file_name, _line_number, _function_name, text = tb[-(level + 2)]
    try:
        tree = ast.parse(text, file_name)
    except SyntaxError:
        return None
    assert tree is not None and isinstance(tree, ast.Module)
    if len(tree.body) == 1:
        stmt = tree.body[0]
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
            if isinstance(expr, ast.Call):
                args = expr.args
                if pos < len(args):
                    res = args[pos]
                    if isinstance(res, ast.Name):
                        return res.id
    return None


def data_to_json(data, subsample_array: bool = True) -> Union[list, dict, int, float]:
    if isinstance(data, tuple):
        # convert to list
        return [data_to_json(elem, subsample_array) for elem in data]
    if isinstance(data, list):
        return [data_to_json(elem, subsample_array) for elem in data]
    elif isinstance(data, dict):
        return {key: data_to_json(data[key], subsample_array) for key in data}
    elif isinstance(data, np.ndarray):
        return ndarray_to_json(data, subsample_array)
    elif isinstance(data, scipy.sparse.csr_matrix):
        return ndarray_to_json(data.toarray(), subsample_array)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        np_array = data.values
        return ndarray_to_json(np_array, subsample_array)
    elif torch is not None and isinstance(data, torch.Tensor):
        np_array = data.detach().numpy()
        return ndarray_to_json(np_array, subsample_array)
    elif isinstance(data, (np.int64, np.int32, np.int16)):  # type: ignore
        return int(data)
    elif isinstance(data, (np.float32, np.float64)):  # type: ignore
        return float(data)
    else:
        return data


def is_empty_dict(val) -> bool:
    return isinstance(val, dict) and len(val) == 0


def dict_without(orig_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in orig_dict:
        return orig_dict
    return {k: v for k, v in orig_dict.items() if k != key}


def json_lookup(ptr, jsn, default=None):
    steps = ptr.split("/")
    sub_jsn = jsn
    for s in steps:
        if s not in sub_jsn:
            return default
        sub_jsn = sub_jsn[s]
    return sub_jsn


def ndarray_to_json(arr: np.ndarray, subsample_array: bool = True) -> Union[list, dict]:
    # sample 10 rows and no limit on columns
    num_subsamples: List[int]
    if subsample_array:
        num_subsamples = [10, np.iinfo(int).max, np.iinfo(int).max]
    else:
        num_subsamples = [
            np.iinfo(int).max,
            np.iinfo(int).max,
            np.iinfo(int).max,
        ]

    def subarray_to_json(indices: Tuple[int, ...]) -> Any:
        if len(indices) == len(arr.shape):
            if isinstance(arr[indices], (bool, int, float, str)):
                return arr[indices]
            elif np.issubdtype(arr.dtype, np.bool_):
                return bool(arr[indices])
            elif np.issubdtype(arr.dtype, np.integer):
                return int(arr[indices])
            elif np.issubdtype(arr.dtype, np.number):
                return float(arr[indices])
            elif arr.dtype.kind in ["U", "S", "O"]:
                return str(arr[indices])
            else:
                raise ValueError(
                    f"Unexpected dtype {arr.dtype}, "
                    f"kind {arr.dtype.kind}, "
                    f"type {type(arr[indices])}."
                )
        else:
            assert len(indices) < len(arr.shape)
            return [
                subarray_to_json(indices + (i,))
                for i in range(
                    min(num_subsamples[len(indices)], arr.shape[len(indices)])
                )
            ]

    return subarray_to_json(())


def split_with_schemas(estimator, all_X, all_y, indices, train_indices=None):
    subset_X, subset_y = _safe_split(estimator, all_X, all_y, indices, train_indices)
    if hasattr(all_X, "json_schema"):
        n_rows = subset_X.shape[0]
        schema = {
            "type": "array",
            "minItems": n_rows,
            "maxItems": n_rows,
            "items": all_X.json_schema["items"],
        }
        lale.datasets.data_schemas.add_schema(subset_X, schema)
    if hasattr(all_y, "json_schema"):
        n_rows = subset_y.shape[0]
        schema = {
            "type": "array",
            "minItems": n_rows,
            "maxItems": n_rows,
            "items": all_y.json_schema["items"],
        }
        lale.datasets.data_schemas.add_schema(subset_y, schema)
    return subset_X, subset_y


def fold_schema(X, y, cv=1, is_classifier=True):
    def fold_schema_aux(data, n_rows):
        orig_schema = lale.datasets.data_schemas._to_schema(data)
        aux_result = {**orig_schema, "minItems": n_rows, "maxItems": n_rows}
        return aux_result

    n_splits = cv if isinstance(cv, int) else cv.get_n_splits()
    try:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    except TypeError:  # raised for Spark dataframes.
        n_samples = X.count() if hasattr(X, "count") else 0

    if n_splits == 1:
        n_rows_fold = n_samples
    elif is_classifier:
        n_classes = len(set(y))
        n_rows_unstratified = (n_samples // n_splits) * (n_splits - 1)
        # in stratified case, fold sizes can differ by up to n_classes
        n_rows_fold = max(1, n_rows_unstratified - n_classes)
    else:
        n_rows_fold = (n_samples // n_splits) * (n_splits - 1)
    schema_X = fold_schema_aux(X, n_rows_fold)
    schema_y = fold_schema_aux(y, n_rows_fold)
    result = {"properties": {"X": schema_X, "y": schema_y}}
    return result


def cross_val_score_track_trials(
    estimator,
    X,
    y=None,
    scoring: Any = accuracy_score,
    cv: Any = 5,
    args_to_scorer: Optional[Dict[str, Any]] = None,
    args_to_cv: Optional[Dict[str, Any]] = None,
    **fit_params,
):
    """
    Use the given estimator to perform fit and predict for splits defined by 'cv' and compute the given score on
    each of the splits.

    Parameters
    ----------

    estimator: A valid sklearn_wrapper estimator
    X: Valid data that works with the estimator
    y: Valid target that works with the estimator
    scoring: string or a scorer object created using
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
        A string from sklearn.metrics.SCORERS.keys() can be used or a scorer created from one of
        sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
        A completely custom scorer object can be created from a python function following the example at
        https://scikit-learn.org/stable/modules/model_evaluation.html
        The metric has to return a scalar value,
    cv: an integer or an object that has a split function as a generator yielding (train, test) splits as arrays of indices.
        Integer value is used as number of folds in sklearn.model_selection.StratifiedKFold, default is 5.
        Note that any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators can be used here.
    args_to_scorer: A dictionary of additional keyword arguments to pass to the scorer.
                Used for cases where the scorer has a signature such as ``scorer(estimator, X, y, **kwargs)``.
    args_to_cv: A dictionary of additional keyword arguments to pass to the split method of cv.
                This is only applicable when cv is not an integer.
    fit_params: Additional parameters that should be passed when calling fit on the estimator
    Returns
    -------
        cv_results: a list of scores corresponding to each cross validation fold
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(cv)

    if args_to_scorer is None:
        args_to_scorer = {}
    if args_to_cv is None:
        args_to_cv = {}
    scorer = check_scoring(estimator, scoring=scoring)
    cv_results: List[float] = []
    log_loss_results = []
    time_results = []
    for train, test in cv.split(X, y, **args_to_cv):
        X_train, y_train = split_with_schemas(estimator, X, y, train)
        X_test, y_test = split_with_schemas(estimator, X, y, test, train)
        start = time.time()
        # Not calling sklearn.base.clone() here, because:
        #  (1) For Lale pipelines, clone() calls the pipeline constructor
        #      with edges=None, so the resulting topology is incorrect.
        #  (2) For Lale individual operators, the fit() method already
        #      clones the impl object, so cloning again is redundant.
        trained = estimator.fit(X_train, y_train, **fit_params)
        score_value = scorer(trained, X_test, y_test, **args_to_scorer)
        execution_time = time.time() - start
        # not all estimators have predict probability
        try:
            y_pred_proba = trained.predict_proba(X_test)
            logloss = log_loss(y_true=y_test, y_pred=y_pred_proba)
            log_loss_results.append(logloss)
        except BaseException:
            logger.debug("Warning, log loss cannot be computed")
        cv_results.append(score_value)
        time_results.append(execution_time)
    result = (
        np.array(cv_results).mean(),
        np.array(log_loss_results).mean(),
        np.array(time_results).mean(),
    )
    return result


def cross_val_score(estimator, X, y=None, scoring: Any = accuracy_score, cv: Any = 5):
    """
    Use the given estimator to perform fit and predict for splits defined by 'cv' and compute the given score on
    each of the splits.

    Parameters
    ----------

    estimator: A valid sklearn_wrapper estimator
    X: Valid data value that works with the estimator
    y: Valid target value that works with the estimator
    scoring: a scorer object from sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
        Default value is accuracy_score.
    cv: an integer or an object that has a split function as a generator yielding (train, test) splits as arrays of indices.
        Integer value is used as number of folds in sklearn.model_selection.StratifiedKFold, default is 5.
        Note that any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators can be used here.

    Returns
    -------
    cv_results: a list of scores corresponding to each cross validation fold
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(cv)

    cv_results = []
    for train, test in cv.split(X, y):
        X_train, y_train = split_with_schemas(estimator, X, y, train)
        X_test, y_test = split_with_schemas(estimator, X, y, test, train)
        trained_estimator = estimator.fit(X_train, y_train)
        predicted_values = trained_estimator.predict(X_test)
        cv_results.append(scoring(y_test, predicted_values))

    return cv_results


def create_individual_op_using_reflection(class_name, operator_name, param_dict):
    instance = None
    if class_name is not None:
        class_name_parts = class_name.split(".")
        assert (
            len(class_name_parts)
        ) > 1, (
            "The class name needs to be fully qualified, i.e. module name + class name"
        )
        module_name = ".".join(class_name_parts[0:-1])
        class_name = class_name_parts[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        if param_dict is None:
            instance = class_()
        else:
            instance = class_(**param_dict)
    return instance


if TYPE_CHECKING:
    import lale.operators


def to_graphviz(
    lale_operator: "lale.operators.Operator",
    ipython_display: bool = True,
    call_depth: int = 1,
):
    import lale.json_operator
    import lale.operators
    import lale.visualize

    if not isinstance(lale_operator, lale.operators.Operator):
        raise TypeError("The input to to_graphviz needs to be a valid LALE operator.")
    jsn = lale.json_operator.to_json(lale_operator, call_depth=call_depth + 1)
    dot = lale.visualize.json_to_graphviz(jsn, ipython_display)
    return dot


def instantiate_from_hyperopt_search_space(obj_hyperparams, new_hyperparams):
    if isinstance(new_hyperparams, dict) and LALE_NESTED_SPACE_KEY in new_hyperparams:
        sub_params = new_hyperparams[LALE_NESTED_SPACE_KEY]

        sub_op = obj_hyperparams
        if isinstance(sub_op, list):
            if len(sub_op) == 1:
                sub_op = sub_op[0]
            else:
                step_index, step_params = list(sub_params)[0]
                if step_index < len(sub_op):
                    sub_op = sub_op[step_index]
                    sub_params = step_params

        return create_instance_from_hyperopt_search_space(sub_op, sub_params)

    elif isinstance(new_hyperparams, (list, tuple)):
        assert isinstance(obj_hyperparams, (list, tuple))
        params_len = len(new_hyperparams)
        assert params_len == len(obj_hyperparams)
        res: Optional[List[Any]] = None

        for i in range(params_len):
            nhi = new_hyperparams[i]
            ohi = obj_hyperparams[i]
            updated_params = instantiate_from_hyperopt_search_space(ohi, nhi)
            if updated_params is not None:
                if res is None:
                    res = list(new_hyperparams)
                res[i] = updated_params
        if res is not None:
            if isinstance(obj_hyperparams, tuple):
                return tuple(res)
            else:
                return res
        # workaround for what seems to be a hyperopt bug
        # where hyperopt returns a tuple even though the
        # hyperopt search space specifies a list
        is_obj_tuple = isinstance(obj_hyperparams, tuple)
        is_new_tuple = isinstance(new_hyperparams, tuple)
        if is_obj_tuple != is_new_tuple:
            if is_obj_tuple:
                return tuple(new_hyperparams)
            else:
                return list(new_hyperparams)
        return None

    elif isinstance(new_hyperparams, dict):
        assert isinstance(obj_hyperparams, dict)

        for k, sub_params in new_hyperparams.items():
            if k in obj_hyperparams:
                sub_op = obj_hyperparams[k]
                updated_params = instantiate_from_hyperopt_search_space(
                    sub_op, sub_params
                )
                if updated_params is not None:
                    new_hyperparams[k] = updated_params
        return None
    else:
        return None


def create_instance_from_hyperopt_search_space(
    lale_object, hyperparams
) -> "lale.operators.Operator":
    """
    Hyperparams is a n-tuple of dictionaries of hyper-parameters, each
    dictionary corresponds to an operator in the pipeline
    """
    # lale_object can either be an individual operator, a pipeline or an operatorchoice
    # Validate that the number of elements in the n-tuple is the same
    # as the number of steps in the current pipeline

    from lale.operators import (
        BasePipeline,
        OperatorChoice,
        PlannedIndividualOp,
        TrainableOperator,
        TrainablePipeline,
    )

    if isinstance(lale_object, PlannedIndividualOp):
        new_hyperparams: Dict[str, Any] = dict_without(hyperparams, "name")
        hps = lale_object.hyperparams()
        if hps:
            obj_hyperparams = dict(hps)
        else:
            obj_hyperparams = {}

        for k, sub_params in new_hyperparams.items():
            if k in obj_hyperparams:
                sub_op = obj_hyperparams[k]
                updated_params = instantiate_from_hyperopt_search_space(
                    sub_op, sub_params
                )
                if updated_params is not None:
                    new_hyperparams[k] = updated_params

        all_hyperparams = {**obj_hyperparams, **new_hyperparams}
        return lale_object(**all_hyperparams)
    elif isinstance(lale_object, BasePipeline):
        steps = lale_object.steps_list()
        if len(hyperparams) != len(steps):
            raise ValueError(
                "The number of steps in the hyper-parameter space does not match the number of steps in the pipeline."
            )
        op_instances = []
        edges = lale_object.edges()
        # op_map:Dict[PlannedOpType, TrainableOperator] = {}
        op_map = {}
        for op_index, sub_params in enumerate(hyperparams):
            sub_op = steps[op_index]
            op_instance = create_instance_from_hyperopt_search_space(sub_op, sub_params)
            assert isinstance(op_instance, TrainableOperator)
            assert (
                isinstance(sub_op, OperatorChoice)
                or sub_op.class_name() == op_instance.class_name()
            ), f"sub_op {sub_op.class_name()}, op_instance {op_instance.class_name()}"
            op_instances.append(op_instance)
            op_map[sub_op] = op_instance

        # trainable_edges:List[Tuple[TrainableOperator, TrainableOperator]]
        try:
            trainable_edges = [(op_map[x], op_map[y]) for (x, y) in edges]
        except KeyError as e:
            raise ValueError(
                "An edge was found with an endpoint that is not a step (" + str(e) + ")"
            ) from e

        return TrainablePipeline(op_instances, trainable_edges, ordered=True)  # type: ignore
    elif isinstance(lale_object, OperatorChoice):
        # Hyperopt search space for an OperatorChoice is generated as a dictionary with a single element
        # corresponding to the choice made, the only key is the index of the step and the value is
        # the params corresponding to that step.
        step_index: int
        choices = lale_object.steps_list()

        if len(choices) == 1:
            step_index = 0
        else:
            step_index_str, hyperparams = list(hyperparams.items())[0]
            step_index = int(step_index_str)
        step_object = choices[step_index]
        return create_instance_from_hyperopt_search_space(step_object, hyperparams)
    else:
        assert False, f"Unknown operator type: {type(lale_object)}"


def find_lale_wrapper(sklearn_obj: Any) -> Optional[Any]:
    """
    :param sklearn_obj: An sklearn compatible object that may have a lale wrapper
    :return: The lale wrapper type, or None if one could not be found
    """
    from .operator_wrapper import get_lale_wrapper_modules

    module_names = get_lale_wrapper_modules()

    class_name = sklearn_obj.__class__.__name__
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        try:
            class_ = getattr(module, class_name)
            return class_
        except AttributeError:
            continue
    return None


def _import_from_sklearn_inplace_helper(
    sklearn_obj, fitted: bool = True, is_nested=False
):
    """
    This method take an object and tries to wrap sklearn objects
    (at the top level or contained within hyperparameters of other
    sklearn objects).
    It will modify the object to add in the appropriate lale wrappers.
    It may also return a wrapper or different object than given.

    :param sklearn_obj: the object that we are going to try and wrap
    :param fitted: should we return a TrainedOperator
    :param is_hyperparams: is this a nested invocation (which allows for returning
    a Trainable operator even if fitted is set to True)
    """

    @overload
    def import_nested_params(
        orig_hyperparams: dict, partial_dict: bool
    ) -> Optional[dict]: ...

    @overload
    def import_nested_params(orig_hyperparams: Any, partial_dict: bool) -> Any: ...

    def import_nested_params(orig_hyperparams: Any, partial_dict: bool = False):
        """
        look through lists/tuples/dictionaries for sklearn compatible objects to import.
        :param orig_hyperparams: the input to recursively look through for sklearn compatible objects
        :param partial_dict: If this is True and the input is a dictionary, the returned dictionary will only have the
        keys with modified values
        :return: Either a modified version of the input or None if nothing was changed
        """
        if isinstance(orig_hyperparams, (tuple, list)):
            new_list: list = []
            list_modified: bool = False
            for e in orig_hyperparams:
                new_e = import_nested_params(e, partial_dict=False)
                if new_e is None:
                    new_list.append(e)
                else:
                    new_list.append(new_e)
                    list_modified = True
            if not list_modified:
                return None
            if isinstance(orig_hyperparams, tuple):
                return tuple(new_list)
            else:
                return new_list
        if isinstance(orig_hyperparams, dict):
            new_dict: dict = {}
            dict_modified: bool = False
            for k, v in orig_hyperparams.items():
                new_v = import_nested_params(v, partial_dict=False)
                if new_v is None:
                    if not partial_dict:
                        new_dict[k] = v
                else:
                    new_dict[k] = new_v
                    dict_modified = True
            if not dict_modified:
                return None
            return new_dict
        if isinstance(orig_hyperparams, object) and hasattr(
            orig_hyperparams, "get_params"
        ):
            newobj = _import_from_sklearn_inplace_helper(
                orig_hyperparams, fitted=fitted, is_nested=True
            )  # allow nested_op to be trainable
            if newobj is orig_hyperparams:
                return None
            return newobj
        return None

    if sklearn_obj is None:
        return None

    if isinstance(sklearn_obj, lale.operators.TrainedIndividualOp):
        # if fitted=False, we may want to return a TrainedIndidivualOp
        return sklearn_obj
    # if the object is a trainable operator, we clean that up
    if isinstance(sklearn_obj, lale.operators.TrainableIndividualOp) and hasattr(
        sklearn_obj, "_trained"
    ):
        if fitted:
            # get rid of the indirection, and just return the trained operator directly
            return sklearn_obj._trained
        else:
            # since we are not supposed to be trained, delete the trained part
            delattr(sklearn_obj, "_trained")  # delete _trained before returning
            return sklearn_obj
    if isinstance(sklearn_obj, lale.operators.Operator):
        if (
            fitted and is_nested or not hasattr(sklearn_obj._impl_instance(), "fit")
        ):  # Operators such as NoOp do not have a fit, so return them as is.
            return sklearn_obj
        if fitted:
            raise ValueError(
                f"""The input pipeline has an operator {sklearn_obj} that is not trained and fitted is set to True,
                please pass fitted=False if you want a trainable pipeline as output."""
            )
        # the lale operator is not trained and fitted=False
        return sklearn_obj

    # special case for FeatureUnion.
    # An alternative would be to (like for sklearn pipeline)
    # create a lale wrapper for the sklearn feature union
    # as a higher order operator
    # and then the special case would be just to throw away the outer wrapper
    # Note that lale union does not currently support weights or other features of feature union.
    if isinstance(sklearn_obj, sklearn.pipeline.FeatureUnion):
        transformer_list = sklearn_obj.transformer_list
        concat_predecessors = [
            _import_from_sklearn_inplace_helper(
                transformer[1], fitted=fitted, is_nested=is_nested
            )
            for transformer in transformer_list
        ]
        return lale.operators.make_union(*concat_predecessors)

    if not hasattr(sklearn_obj, "get_params"):
        # if it does not have a get_params method,
        # then we just return it without trying to wrap it
        return sklearn_obj

    class_ = find_lale_wrapper(sklearn_obj)
    if not class_:
        return sklearn_obj  # Return the original object

    # next, we need to figure out what the right hyperparameters are
    orig_hyperparams = sklearn_obj.get_params(deep=False)

    hyperparams = import_nested_params(orig_hyperparams, partial_dict=True)
    if hyperparams:
        # if we have updated any of the hyperparameters then we modify them in the actual sklearn object
        try:
            new_obj = sklearn_obj.set_params(**hyperparams)
            if new_obj is not None:
                sklearn_obj = new_obj
        except NotImplementedError:
            # if the set_params method does not work, then do our best
            pass

        all_new_hyperparams = {**orig_hyperparams, **hyperparams}
    else:
        all_new_hyperparams = orig_hyperparams

    # now, we get the lale operator for the wrapper, with the corresponding hyperparameters
    if not fitted:  # If fitted is False, we do not want to return a Trained operator.
        lale_op_obj_base = class_
    else:
        lale_op_obj_base = lale.operators.TrainedIndividualOp(
            class_._name,
            class_._impl,
            class_._schemas,
            None,
            _lale_trained=True,
        )

    lale_op_obj = lale_op_obj_base(**all_new_hyperparams)
    from lale.lib.sklearn import Pipeline as LaleSKPipelineWrapper

    # If this is a scklearn pipeline, then we want to discard the outer wrapper
    # and just return a lale pipeline
    if isinstance(lale_op_obj, LaleSKPipelineWrapper):  # type: ignore
        return lale_op_obj.shallow_impl._pipeline

    # at this point, the object's hyper-parameters are modified as needed
    # and our wrapper is initialized with the correct hyperparameters.
    # Now we need to replace the wrapper impl with our (possibly modified)
    # sklearn object
    cl_shallow_impl = lale_op_obj.shallow_impl

    if hasattr(cl_shallow_impl, "_wrapped_model"):
        cl_shallow_impl._wrapped_model = sklearn_obj
    else:
        lale_op_obj._impl = sklearn_obj
        lale_op_obj._impl_class_ = sklearn_obj.__class__

    return lale_op_obj


def import_from_sklearn(sklearn_obj: Any, fitted: bool = True, in_place: bool = False):
    """
    This method take an object and tries to wrap sklearn objects
    (at the top level or contained within hyperparameters of other
    sklearn objects).
    It will modify the object to add in the appropriate lale wrappers.
    It may also return a wrapper or different object than given.

    :param sklearn_obj: the object that we are going to try and wrap
    :param fitted: should we return a TrainedOperator
    :param in_place: should we try to mutate what we can in place, or should we
           aggressively deepcopy everything
    :return: The wrapped object (or the input object if we could not wrap it)
    """
    obj = sklearn_obj
    if in_place:
        obj = sklearn_obj
    else:
        obj = copy.deepcopy(sklearn_obj)
    return _import_from_sklearn_inplace_helper(obj, fitted=fitted, is_nested=False)


def import_from_sklearn_pipeline(sklearn_pipeline: Any, fitted: bool = True):
    """
    Note: Same as import_from_sklearn.  This alternative name exists for backwards compatibility.

    This method take an object and tries to wrap sklearn objects
    (at the top level or contained within hyperparameters of other
    sklearn objects).
    It will modify the object to add in the appropriate lale wrappers.
    It may also return a wrapper or different object than given.

    :param sklearn_pipeline: the object that we are going to try and wrap
    :param fitted: should we return a TrainedOperator
    :return: The wrapped object (or the input object if we could not wrap it)

    """
    op = import_from_sklearn(sklearn_pipeline, fitted=fitted, in_place=False)

    from typing import cast

    from lale.operators import TrainableOperator

    # simplify using the returned value in the common case
    return cast(TrainableOperator, op)


class val_wrapper:
    """This is used to wrap values that cause problems for hyper-optimizer backends
    lale will unwrap these when given them as the value of a hyper-parameter"""

    def __init__(self, base):
        self._base = base

    def unwrap_self(self):
        return self._base

    @classmethod
    def unwrap(cls, obj):
        if isinstance(obj, cls):
            return cls.unwrap(obj.unwrap_self())
        else:
            return obj


def append_batch(data, batch_data):
    if data is None:
        return batch_data
    elif isinstance(data, np.ndarray):
        if isinstance(batch_data, np.ndarray):
            if len(data.shape) == 1 and len(batch_data.shape) == 1:
                return np.concatenate([data, batch_data])
            else:
                return np.vstack((data, batch_data))
    elif isinstance(data, tuple):
        X, y = data
        if isinstance(batch_data, tuple):
            batch_X, batch_y = batch_data
            X = append_batch(X, batch_X)
            y = append_batch(y, batch_y)
            return X, y
    elif torch is not None and isinstance(data, torch.Tensor):
        if isinstance(batch_data, torch.Tensor):
            return torch.cat((data, batch_data))
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return pd.concat([data, batch_data], axis=0)
    try:
        import h5py

        if isinstance(data, h5py.File):
            if isinstance(batch_data, tuple):
                batch_X, batch_y = batch_data
    except ModuleNotFoundError:
        pass

    raise ValueError(
        f"{type(data)} is unsupported. Supported types are np.ndarray, torch.Tensor and h5py file"
    )


def create_data_loader(
    X: Any,
    y: Any = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """A function that takes a dataset as input and outputs a Pytorch dataloader.

    Parameters
    ----------
    X : Input data.
        The formats supported are Pandas DataFrame, Numpy array,
        a sparse matrix, torch.tensor, torch.utils.data.Dataset, path to a HDF5 file,
        lale.util.batch_data_dictionary_dataset.BatchDataDict,
        a Python dictionary of the format `{"dataset": torch.utils.data.Dataset,
        "collate_fn":collate_fn for torch.utils.data.DataLoader}`
    y : Labels., optional
        Supported formats are Numpy array or Pandas series, by default None
    batch_size : int, optional
        Number of samples in each batch, by default 1
    num_workers : int, optional
        Number of workers used by the data loader, by default 0
    shuffle: boolean, optional, default True
        Whether to use SequentialSampler or RandomSampler for creating batches

    Returns
    -------
    torch.utils.data.DataLoader

    Raises
    ------
    TypeError
        Raises a TypeError if the input format is not supported.
    """
    assert torch is not None
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    from lale.util.batch_data_dictionary_dataset import BatchDataDict
    from lale.util.hdf5_to_torch_dataset import HDF5TorchDataset
    from lale.util.numpy_torch_dataset import NumpyTorchDataset, numpy_collate_fn
    from lale.util.pandas_torch_dataset import PandasTorchDataset, pandas_collate_fn

    collate_fn = None
    worker_init_fn = None

    if isinstance(X, Dataset) and not isinstance(X, BatchDataDict):
        dataset = X
    elif isinstance(X, pd.DataFrame):
        dataset = PandasTorchDataset(X, y)
        collate_fn = pandas_collate_fn
    elif isinstance(X, scipy.sparse.csr_matrix):
        # unfortunately, NumpyTorchDataset won't accept a subclass of np.ndarray
        X = X.toarray()  # type: ignore
        if isinstance(y, lale.datasets.data_schemas.NDArrayWithSchema):
            y = y.view(np.ndarray)
        dataset = NumpyTorchDataset(X, y)
        collate_fn = numpy_collate_fn
    elif isinstance(X, np.ndarray):
        # unfortunately, NumpyTorchDataset won't accept a subclass of np.ndarray
        if isinstance(X, lale.datasets.data_schemas.NDArrayWithSchema):
            X = X.view(np.ndarray)
        if isinstance(y, lale.datasets.data_schemas.NDArrayWithSchema):
            y = y.view(np.ndarray)
        dataset = NumpyTorchDataset(X, y)
        collate_fn = numpy_collate_fn
    elif isinstance(X, str):  # Assume that this is path to hdf5 file
        dataset = HDF5TorchDataset(X)
    elif isinstance(X, BatchDataDict):
        dataset = X

        def my_collate_fn(batch):
            return batch[
                0
            ]  # because BatchDataDict's get_item returns a batch, so no collate is required.

        return DataLoader(
            dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=shuffle
        )
    elif isinstance(X, dict):  # Assumed that it is data indexed by batch number
        if "dataset" in X:
            dataset = X["dataset"]
            collate_fn = X.get("collate_fn", None)
            worker_init_fn = getattr(dataset, "worker_init_fn", None)
        else:
            return [X]
    elif isinstance(X, torch.Tensor) and y is not None:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        dataset = TensorDataset(X, y)
    elif isinstance(X, torch.Tensor):
        dataset = TensorDataset(X)
    else:
        raise TypeError(
            f"Can not create a data loader for a dataset with type {type(X)}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=shuffle,
    )


def write_batch_output_to_file(
    file_obj,
    file_path,
    total_len,
    batch_idx,
    batch_X,
    batch_y,
    batch_out_X,
    batch_out_y,
):
    if file_obj is None and file_path is None:
        raise ValueError("Only one of the file object or file path can be None.")
    if file_obj is None:
        import h5py

        file_obj = h5py.File(file_path, "w")
        # estimate the size of the dataset based on the first batch output size
        transform_ratio = int(len(batch_out_X) / len(batch_X))
        if len(batch_out_X.shape) == 1:
            h5_data_shape = (transform_ratio * total_len,)
        elif len(batch_out_X.shape) == 2:
            h5_data_shape = (transform_ratio * total_len, batch_out_X.shape[1])
        elif len(batch_out_X.shape) == 3:
            h5_data_shape = (
                transform_ratio * total_len,
                batch_out_X.shape[1],
                batch_out_X.shape[2],
            )
        else:
            raise ValueError(
                "batch_out_X is expected to be a 1-d, 2-d or 3-d array. Any other data types are not handled."
            )
        dataset = file_obj.create_dataset(
            name="X", shape=h5_data_shape, chunks=True, compression="gzip"
        )
        if batch_out_y is None and batch_y is not None:
            batch_out_y = batch_y
        if batch_out_y is not None:
            if len(batch_out_y.shape) == 1:
                h5_labels_shape = (transform_ratio * total_len,)
            elif len(batch_out_y.shape) == 2:
                h5_labels_shape = (transform_ratio * total_len, batch_out_y.shape[1])
            else:
                raise ValueError(
                    "batch_out_y is expected to be a 1-d or 2-d array. Any other data types are not handled."
                )
            dataset = file_obj.create_dataset(
                name="y", shape=h5_labels_shape, chunks=True, compression="gzip"
            )
    dataset = file_obj["X"]
    dataset[batch_idx * len(batch_out_X) : (batch_idx + 1) * len(batch_out_X)] = (
        batch_out_X
    )
    if batch_out_y is not None or batch_y is not None:
        labels = file_obj["y"]
        if batch_out_y is not None:
            labels[
                batch_idx * len(batch_out_y) : (batch_idx + 1) * len(batch_out_y)
            ] = batch_out_y
        else:
            labels[batch_idx * len(batch_y) : (batch_idx + 1) * len(batch_y)] = batch_y
    return file_obj


def add_missing_values(orig_X, missing_rate=0.1, seed=None):
    # see scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html
    n_samples, n_features = orig_X.shape
    n_missing_samples = int(n_samples * missing_rate)
    if seed is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[:n_missing_samples] = True
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    missing_X = orig_X.copy()
    if isinstance(missing_X, np.ndarray):
        missing_X[missing_samples, missing_features] = np.nan
    else:
        assert isinstance(missing_X, pd.DataFrame)
        i_missing_sample = 0
        for i_sample in range(n_samples):
            if missing_samples[i_sample]:
                i_feature = missing_features[i_missing_sample]
                i_missing_sample += 1
                missing_X.iloc[i_sample, i_feature] = np.nan
    return missing_X


# helpers for manipulating (extended) sklearn style paths.
# documentation of the path format is part of the operators module docstring


def partition_sklearn_params(
    d: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    sub_parts: Dict[str, Dict[str, Any]] = {}
    main_parts: Dict[str, Any] = {}

    for k, v in d.items():
        ks = k.split("__", 1)
        if len(ks) == 1:
            assert k not in main_parts
            main_parts[k] = v
        else:
            assert len(ks) == 2
            bucket: Dict[str, Any] = {}
            group: str = ks[0]
            param: str = ks[1]
            if group in sub_parts:
                bucket = sub_parts[group]
            else:
                sub_parts[group] = bucket
            assert param not in bucket
            bucket[param] = v
    return (main_parts, sub_parts)


def partition_sklearn_choice_params(d: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    discriminant_value: int = -1
    choice_parts: Dict[str, Any] = {}

    for k, v in d.items():
        if k == discriminant_name:
            assert discriminant_value == -1
            discriminant_value = int(v)
        else:
            k_rest = unnest_choice(k)
            choice_parts[k_rest] = v
    assert discriminant_value != -1
    return (discriminant_value, choice_parts)


DUMMY_SEARCH_SPACE_GRID_PARAM_NAME: str = "$"
discriminant_name: str = "?"
choice_prefix: str = "?"
structure_type_name: str = "#"
structure_type_list: str = "list"
structure_type_tuple: str = "tuple"
structure_type_dict: str = "dict"


def get_name_and_index(name: str) -> Tuple[str, int]:
    """given a name of the form "name@i", returns (name, i)
    if given a name of the form "name", returns (name, 0)
    """
    splits = name.split("@", 1)
    if len(splits) == 1:
        return splits[0], 0
    else:
        return splits[0], int(splits[1])


def make_degen_indexed_name(name, index):
    return f"{name}@{index}"


def make_indexed_name(name, index):
    if index == 0:
        return name
    else:
        return f"{name}@{index}"


def make_array_index_name(index, is_tuple: bool = False):
    sep = "##" if is_tuple else "#"
    return f"{sep}{str(index)}"


def is_numeric_structure(structure_type: str):
    if structure_type in ["list", "tuple"]:
        return True
    elif structure_type == "dict":
        return False
    else:
        assert False, f"Unknown structure type {structure_type} found"


V = TypeVar("V")


def nest_HPparam(name: str, key: str):
    if key == DUMMY_SEARCH_SPACE_GRID_PARAM_NAME:
        # we can get rid of the dummy now, since we have a name for it
        return name
    return name + "__" + key


def nest_HPparams(name: str, grid: Mapping[str, V]) -> Dict[str, V]:
    return {(nest_HPparam(name, k)): v for k, v in grid.items()}


def nest_all_HPparams(
    name: str, grids: Iterable[Mapping[str, V]]
) -> List[Dict[str, V]]:
    """Given the name of an operator in a pipeline, this transforms every key(parameter name) in the grids
    to use the operator name as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_HPparams(name, grid) for grid in grids]


def nest_choice_HPparam(key: str):
    return choice_prefix + key


def nest_choice_HPparams(grid: Mapping[str, V]) -> Dict[str, V]:
    return {(nest_choice_HPparam(k)): v for k, v in grid.items()}


def nest_choice_all_HPparams(grids: Iterable[Mapping[str, V]]) -> List[Dict[str, V]]:
    """this transforms every key(parameter name) in the grids
    to be nested under a choice, using a ? as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_choice_HPparams(grid) for grid in grids]


def unnest_choice(k: str) -> str:
    assert k.startswith(choice_prefix)
    return k[len(choice_prefix) :]


def unnest_HPparams(k: str) -> List[str]:
    return k.split("__")


def are_hyperparameters_equal(hyperparam1, hyperparam2):
    if isinstance(
        hyperparam1, np.ndarray
    ):  # hyperparam2 is from schema default, so it may not always be an array
        return np.all(hyperparam1 == hyperparam2)
    else:
        return hyperparam1 == hyperparam2


def _is_ast_subscript(expr):
    return isinstance(expr, ast.Subscript)


def _is_ast_attribute(expr):
    return isinstance(expr, ast.Attribute)


def _is_ast_constant(expr):
    return isinstance(expr, ast.Constant)


def _is_ast_subs_or_attr(expr):
    return isinstance(expr, (ast.Subscript, ast.Attribute))


def _is_ast_call(expr):
    return isinstance(expr, ast.Call)


def _is_ast_name(expr):
    return isinstance(expr, ast.Name)


def _ast_func_id(expr):
    if isinstance(expr, ast.Name):
        return expr.id
    else:
        raise ValueError("function name expected")


def _is_df(df):
    return _is_pandas_df(df) or _is_spark_df(df)


def _is_pandas_series(df):
    return isinstance(df, pd.Series)


def _is_pandas_df(df):
    return isinstance(df, pd.DataFrame)


def _is_pandas(df):
    return isinstance(df, (pd.Series, pd.DataFrame))


def _is_spark_df(df):
    if spark_installed:
        return isinstance(df, lale.datasets.data_schemas.SparkDataFrameWithIndex)
    else:
        return False


def _is_spark_df_without_index(df):
    return spark_df is not None and isinstance(df, spark_df) and not _is_spark_df(df)


def _ensure_pandas(df) -> pd.DataFrame:
    if _is_spark_df(df):
        return df.toPandas()
    assert _is_pandas(df), type(df)
    return df


def _get_subscript_value(subscript_expr):
    if isinstance(subscript_expr.slice, ast.Constant):  # for Python 3.9
        subscript_value = subscript_expr.slice.value
    else:
        subscript_value = subscript_expr.slice.value.s  # type: ignore
    return subscript_value


class GenSym:
    def __init__(self, names: Set[str]):
        self._names = names

    def __call__(self, prefix):
        if prefix in self._names:
            suffix = 0
            while True:
                result = f"{prefix}_{suffix}"
                if result not in self._names:
                    break
                suffix += 1
        else:
            result = prefix
        self._names |= {result}
        return result


def _should_force_estimator():
    from packaging import version

    import lale.operators

    return lale.operators.sklearn_version >= version.Version("1.5")


def get_sklearn_estimator_name() -> str:
    """Some higher order sklearn operators changed the name of the nested estimatator in later versions.
    This returns the appropriate version dependent paramater name
    """
    from packaging import version

    import lale.operators

    if lale.operators.sklearn_version < version.Version("1.2"):
        return "base_estimator"
    else:
        return "estimator"


def with_fixed_estimator_name(**kwargs):
    """Some higher order sklearn operators changed the name of the nested estimator in later versions.
    This fixes up the arguments, renaming estimator and base_estimator appropriately.
    """

    if "base_estimator" in kwargs or "estimator" in kwargs:
        from packaging import version

        import lale.operators

        if lale.operators.sklearn_version < version.Version("1.2"):
            return {
                "base_estimator" if k == "estimator" else k: v
                for k, v in kwargs.items()
            }
        else:
            return {
                "estimator" if k == "base_estimator" else k: v
                for k, v in kwargs.items()
            }

    return kwargs


def get_estimator_param_name_from_hyperparams(hyperparams):
    if _should_force_estimator():
        return "estimator"
    be = hyperparams.get("base_estimator", "deprecated")
    if be == "deprecated" or (be is None and "estimator" in hyperparams):
        return "estimator"
    else:
        return "base_estimator"
