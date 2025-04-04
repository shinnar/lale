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

"""Classes for Lale operators including individual operators, pipelines, and operator choice.

This module declares several functions for constructing individual
operators, pipelines, and operator choices.

- Functions `make_pipeline`_ and `Pipeline`_ compose linear sequential
  pipelines, where each step has an edge to the next step. Instead of
  these functions you can also use the `>>` combinator.

- Functions `make_union_no_concat`_ and `make_union`_ compose
  pipelines that operate over the same data without edges between
  their steps. Instead of these functions you can also use the `&`
  combinator.

- Function `make_choice` creates an operator choice. Instead of this
  function you can also use the `|` combinator.

- Function `make_pipeline_graph`_ creates a pipeline from
  steps and edges, thus supporting any arbitrary acyclic directed
  graph topology.

- Function `make_operator`_ creates an individual Lale operator from a
  schema and an implementation class or object. This is called for each
  of the operators in module lale.lib when it is being imported.

- Functions `get_available_operators`_, `get_available_estimators`_,
  and `get_available_transformers`_ return lists of individual
  operators previously registered by `make_operator`.

.. _make_operator: lale.operators.html#lale.operators.make_operator
.. _get_available_operators: lale.operators.html#lale.operators.get_available_operators
.. _get_available_estimators: lale.operators.html#lale.operators.get_available_estimators
.. _get_available_transformers: lale.operators.html#lale.operators.get_available_transformers
.. _make_pipeline_graph: lale.operators.html#lale.operators.make_pipeline_graph
.. _make_pipeline: lale.operators.html#lale.operators.make_pipeline
.. _Pipeline: Lale.Operators.Html#Lale.Operators.Pipeline
.. _make_union_no_concat: lale.operators.html#lale.operators.make_union_no_concat
.. _make_union: lale.operators.html#lale.operators.make_union
.. _make_choice: lale.operators.html#lale.operators.make_choice

The root of the hierarchy is the abstract class Operator_, all other
Lale operators inherit from this class, either directly or indirectly.

- The abstract classes Operator_, PlannedOperator_,
  TrainableOperator_, and TrainedOperator_ correspond to lifecycle
  states.

- The concrete classes IndividualOp_, PlannedIndividualOp_,
  TrainableIndividualOp_, and TrainedIndividualOp_ inherit from the
  corresponding abstract operator classes and encapsulate
  implementations of individual operators from machine-learning
  libraries such as scikit-learn.

- The concrete classes BasePipeline_, PlannedPipeline_,
  TrainablePipeline_, and TrainedPipeline_ inherit from the
  corresponding abstract operator classes and represent directed
  acyclic graphs of operators. The steps of a pipeline can be any
  operators, including individual operators, other pipelines, or
  operator choices, whose lifecycle state is at least that of the
  pipeline.

- The concrete class OperatorChoice_ represents a planned operator
  that offers a choice for automated algorithm selection. The steps of
  a choice can be any planned operators, including individual
  operators, pipelines, or other operator choices.

The following picture illustrates the core operator class hierarchy.

.. image:: ../../docs/img/operator_classes.png
  :alt: operators class hierarchy

.. _BasePipeline: lale.operators.html#lale.operators.BasePipeline
.. _IndividualOp: lale.operators.html#lale.operators.IndividualOp
.. _Operator: lale.operators.html#lale.operators.Operator
.. _OperatorChoice: lale.operators.html#lale.operators.OperatorChoice
.. _PlannedIndividualOp: lale.operators.html#lale.operators.PlannedIndividualOp
.. _PlannedOperator: lale.operators.html#lale.operators.PlannedOperator
.. _PlannedPipeline: lale.operators.html#lale.operators.PlannedPipeline
.. _TrainableIndividualOp: lale.operators.html#lale.operators.TrainableIndividualOp
.. _TrainableOperator: lale.operators.html#lale.operators.TrainableOperator
.. _TrainablePipeline: lale.operators.html#lale.operators.TrainablePipeline
.. _TrainedIndividualOp: lale.operators.html#lale.operators.TrainedIndividualOp
.. _TrainedOperator: lale.operators.html#lale.operators.TrainedOperator
.. _TrainedPipeline: lale.operators.html#lale.operators.TrainedPipeline

scikit-learn compatibility:
---------------------------

Lale operators attempt to behave like reasonable sckit-learn operators when possible.
In particular, operators support:

- get_params to return the hyperparameter settings for an operator.
- set_params for updating them (in-place).  This is only supported by TrainableIndividualOps and Pipelines.
  Note that while set_params is supported for
  compatibility, but its use is not encouraged, since it mutates the operator in-place.
  Instead, we recommend using with_params, a functional alternative that is supported by all
  operators.  It returns a new operator with updated parameters.
- sklearn.base.clone works for Lale operators, cloning them as expected.
  Note that cloning a TrainedOperator will return a TrainableOperator, since
  the cloned version does not have the result of training.

There also some known differences (that we are not currently planning on changing):

- Lale operators do not inherit from any sklearn base class.
- The Operator class constructors do not explicitly declare their set of hyperparameters.
  However, the do implement get_params, (just not using sklearn style reflection).

There may also be other incompatibilities: our testing currently focuses on ensuring that clone works.

parameter path format:
^^^^^^^^^^^^^^^^^^^^^^

scikit-learn uses a simple addressing scheme to refer to nested hyperparameter: `name__param` refers to the
`param` hyperparameter nested under the `name` object.
Since lale supports richer structures, we conservatively extend this scheme as follows:

* `__` : separates nested components (as-in sklearn).
* `?` : is the discriminant (choice made) for a choice.
* `?` : is also a prefix for the nested parts of the chosen branch.
* `x@n` : In a pipeline, if multiple components have identical names,
  everything but the first are suffixed with a number (starting with 1)
  indicating which one we are talking about.
  For example, given `(x >> y >> x)`, we would treat this much the same as
  `(x >> y >> x@1)`.
* `$` : is used in the rare case that sklearn would expect the key of an object,
  but we allow (and have) a non-object schema.  In that case, $ is used as the key.
  This should only happen at the top level, since nested occurrences should be removed.
* `#` : is a structure indicator, and the value should be one of 'list', 'tuple', or 'dict'.
* `n` : is used to represent the nth component in an array or tuple.


"""

import copy
import difflib
import enum as enumeration
import importlib
import inspect
import itertools
import logging
import os
import warnings
from abc import abstractmethod
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import jsonschema
import pandas as pd
import sklearn
import sklearn.base
from packaging import version
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if

import lale.datasets.data_schemas
import lale.json_operator
import lale.pretty_print
from lale import schema2enums as enum_gen
from lale.datasets.data_schemas import (
    NDArrayWithSchema,
    _to_schema,
    add_schema,
    strip_schema,
)
from lale.helpers import (
    append_batch,
    are_hyperparameters_equal,
    assignee_name,
    astype_type,
    fold_schema,
    get_name_and_index,
    is_empty_dict,
    is_numeric_structure,
    make_degen_indexed_name,
    make_indexed_name,
    nest_HPparams,
    partition_sklearn_choice_params,
    partition_sklearn_params,
    structure_type_name,
    to_graphviz,
    val_wrapper,
)
from lale.json_operator import JSON_TYPE
from lale.schemas import Schema
from lale.search.PGO import remove_defaults_dict
from lale.type_checking import (
    SubschemaError,
    get_default_schema,
    has_data_constraints,
    is_subschema,
    join_schemas,
    replace_data_constraints,
    validate_is_schema,
    validate_method,
    validate_schema,
    validate_schema_directly,
)
from lale.util.VisitorMeta import AbstractVisitorMeta

sklearn_version = version.parse(getattr(sklearn, "__version__"))

logger = logging.getLogger(__name__)

_LALE_SKL_PIPELINE = "lale.lib.sklearn.pipeline._PipelineImpl"


def _impl_has(attr):
    return lambda self: (hasattr(self._impl, attr))


def _trained_impl_has(attr):
    def f(self):
        op = getattr(self, "_trained", self)
        if op is None:
            return False
        return _impl_has(attr)(op)

    return f


def _final_impl_has(attr):
    def f(self):
        estimator = self._final_individual_op
        if estimator is not None:
            return _impl_has(attr)(estimator)
        else:
            return False

    return f


def _final_trained_impl_has(attr):
    def f(self):
        estimator = self._final_individual_op
        if estimator is not None:
            return _trained_impl_has(attr)(estimator)
        else:
            return False

    return f


_combinators_docstrings = """
    Methods
    -------

    step_1 >> step_2 -> PlannedPipeline
        Pipe combinator, create two-step pipeline with edge from step_1 to step_2.

        If step_1 is a pipeline, create edges from all of its sinks.
        If step_2 is a pipeline, create edges to all of its sources.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The origin of the edge(s).
        step_2 : Operator
            The destination of the edge(s).

        Returns
        ^^^^^^^
        BasePipeline
            Pipeline with edge from step_1 to step_2.

    step_1 & step_2 -> PlannedPipeline
        And combinator, create two-step pipeline without an edge between step_1 and step_2.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The first step.
        step_2 : Operator
            The second step.

        Returns
        ^^^^^^^
        BasePipeline
            Pipeline without any additional edges beyond those already inside of step_1 or step_2.

    step_1 | step_2 -> OperatorChoice
        Or combinator, create operator choice between step_1 and step_2.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The first step.
        step_2 : Operator
            The second step.

        Returns
        ^^^^^^^
        OperatorChoice
            Algorithmic coice between step_1 or step_2."""


class Operator(metaclass=AbstractVisitorMeta):
    """Abstract base class for all Lale operators.

    Pipelines and individual operators extend this."""

    _name: str

    def __and__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline":
        return make_union_no_concat(self, other)

    def __rand__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline":
        return make_union_no_concat(other, self)

    def __rshift__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline":
        return make_pipeline(self, other)

    def __rrshift__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline":
        return make_pipeline(other, self)

    def __or__(self, other: Union[Any, "Operator"]) -> "OperatorChoice":
        return make_choice(self, other)

    def __ror__(self, other: Union[Any, "Operator"]) -> "OperatorChoice":
        return make_choice(other, self)

    def name(self) -> str:
        """Get the name of this operator instance."""
        return self._name

    def _set_name(self, name: str):
        """Set the name of this operator instance."""
        self._name = name

    def class_name(self) -> str:
        """Fully qualified Python class name of this operator."""
        cls = self.__class__
        return cls.__module__ + "." + cls.__name__  # type: ignore

    @abstractmethod
    def validate_schema(self, X: Any, y: Any = None):
        """Validate that X and y are valid with respect to the input schema of this operator.

        Parameters
        ----------
        X :
            Features.
        y :
            Target class labels or None for unsupervised operators.

        Raises
        ------
        ValueError
            If X or y are invalid as inputs."""
        pass

    @abstractmethod
    def transform_schema(self, s_X: JSON_TYPE) -> JSON_TYPE:
        """Return the output schema given the input schema.

        Parameters
        ----------
        s_X :
            Input dataset or schema.

        Returns
        -------
        JSON schema
            Schema of the output data given the input data schema."""
        pass

    @abstractmethod
    def input_schema_fit(self) -> JSON_TYPE:
        """Input schema for the fit method."""
        pass

    def to_json(self) -> JSON_TYPE:
        """Returns the JSON representation of the operator.

        Returns
        -------
        JSON document
            JSON representation that describes this operator and is valid with respect to lale.json_operator.SCHEMA.
        """
        return lale.json_operator.to_json(self, call_depth=2)

    def get_forwards(self) -> Union[bool, List[str]]:
        """Returns the list of attributes (methods/properties)
        the schema has asked to be forwarded.  A boolean value is a blanket
        opt-in or out of forwarding
        """
        return False

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """For scikit-learn compatibility"""
        pass

    def visualize(self, ipython_display: bool = True):
        """Visualize the operator using graphviz (use in a notebook).

        Parameters
        ----------
        ipython_display : bool, default True
            If True, proactively ask Jupyter to render the graph.
            Otherwise, the graph will only be rendered when visualize()
            was called in the last statement in a notebook cell.

        Returns
        -------
        Digraph
            Digraph object from the graphviz package.
        """
        return to_graphviz(self, ipython_display, call_depth=2)

    @overload
    def pretty_print(
        self,
        *,
        show_imports: bool = True,
        combinators: bool = True,
        assign_nested: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        astype: astype_type = "lale",
        ipython_display: Literal[False] = False,
    ) -> str: ...

    @overload
    def pretty_print(
        self,
        *,
        show_imports: bool = True,
        combinators: bool = True,
        assign_nested: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        astype: astype_type = "lale",
        ipython_display: Union[bool, Literal["input"]] = False,
    ) -> Optional[str]: ...

    def pretty_print(
        self,
        *,
        show_imports: bool = True,
        combinators: bool = True,
        assign_nested: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        astype: astype_type = "lale",
        ipython_display: Union[bool, Literal["input"]] = False,
    ) -> Optional[str]:
        """Returns the Python source code representation of the operator.

        Parameters
        ----------
        show_imports : bool, default True

            Whether to include import statements in the pretty-printed code.

        combinators : bool, default True

            If True, pretty-print with combinators (`>>`, `|`, `&`). Otherwise, pretty-print with functions (`make_pipeline`, `make_choice`, `make_union`) instead. Always False when astype is 'sklearn'.

        assign_nested : bool, default True

            If True, then nested operators, such as the base estimator for an ensemble, get assigned to fresh intermediate variables if configured with non-trivial arguments of their own.

        customize_schema : bool, default False

            If True, then individual operators whose schema differs from the lale.lib version of the operator will be printed with calls to `customize_schema` that reproduce this difference.

        astype : union type, default 'lale'

            - 'lale'

              Use `lale.operators.make_pipeline` and `lale.operators.make_union` when pretty-printing wth functions.

            - 'sklearn'

              Set combinators to False and use `sklearn.pipeline.make_pipeline` and `sklearn.pipeline.make_union` for pretty-printed functions.

        ipython_display : union type, default False

            - False

              Return the pretty-printed code as a plain old Python string.

            - True:

              Pretty-print in notebook cell output with syntax highlighting.

            - 'input'

              Create a new notebook cell with pretty-printed code as input.

        Returns
        -------
        str or None
            If called with ipython_display=False, return pretty-printed Python source code as a Python string.
        """
        result = lale.pretty_print.to_string(
            self,
            show_imports=show_imports,
            combinators=combinators,
            customize_schema=customize_schema,
            assign_nested=assign_nested,
            astype=astype,
            call_depth=2,
        )
        if ipython_display is False:
            return result
        elif ipython_display == "input":
            import IPython.core

            ipython = IPython.core.getipython.get_ipython()
            comment = "# generated by pretty_print(ipython_display='input') from previous cell\n"
            ipython.set_next_input(comment + result, replace=False)
            return None
        else:
            assert ipython_display in [True, "output"]
            import IPython.display

            markdown = IPython.display.Markdown(f"```python\n{result}\n```")
            IPython.display.display(markdown)
            return None

    @overload
    def diff(
        self,
        other: "Operator",
        show_imports: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        ipython_display: Literal[False] = False,
    ) -> str: ...

    @overload
    def diff(
        self,
        other: "Operator",
        show_imports: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        ipython_display: bool = False,
    ) -> Optional[str]: ...

    def diff(
        self,
        other: "Operator",
        show_imports: bool = True,
        customize_schema: bool = False,  # pylint:disable=redefined-outer-name
        ipython_display: bool = False,
    ) -> Optional[str]:
        """Displays a diff between this operator and the given other operator.

        Parameters
        ----------
        other: Operator
            Operator to diff against

        show_imports : bool, default True
            Whether to include import statements in the pretty-printed code.

        customize_schema : bool, default False
            If True, then individual operators whose schema differs from the lale.lib version of the operator will be printed with calls to `customize_schema` that reproduce this difference.

        ipython_display : bool, default False
            If True, will display Markdown-formatted diff string in Jupyter notebook.
            If False, returns pretty-printing diff as Python string.

        Returns
        -------
        str or None
            If called with ipython_display=False, return pretty-printed diff as a Python string.
        """

        self_str = self.pretty_print(
            customize_schema=customize_schema,
            show_imports=show_imports,
            ipython_display=False,
        )
        self_lines = self_str.splitlines()

        other_str = other.pretty_print(
            customize_schema=customize_schema,
            show_imports=show_imports,
            ipython_display=False,
        )
        other_lines = other_str.splitlines()

        differ = difflib.Differ()
        compare = differ.compare(self_lines, other_lines)

        compare_str = "\n".join(compare)
        if not ipython_display:
            return compare_str
        else:
            import IPython.display

            markdown = IPython.display.Markdown(f"```diff\n{compare_str}\n```")
            IPython.display.display(markdown)
            return None

    @abstractmethod
    def _has_same_impl(self, other: "Operator") -> bool:
        """Checks if the type of the operator implementations are compatible"""
        pass

    @abstractmethod
    def is_supervised(self) -> bool:
        """Checks if this operator needs labeled data for learning.

        Returns
        -------
        bool
            True if the fit method requires a y argument.
        """
        pass

    @abstractmethod
    def is_classifier(self) -> bool:
        """Checks if this operator is a clasifier.

        Returns
        -------
        bool
            True if the classifier tag is set.
        """
        pass

    def is_frozen_trainable(self) -> bool:
        """Return true if all hyperparameters are bound, in other words,
        search spaces contain no free hyperparameters to be tuned.
        """
        return False

    def is_frozen_trained(self) -> bool:
        """Return true if all learnable coefficients are bound, in other
        words, there are no free parameters to be learned by fit.
        """
        return False

    @property
    def _final_individual_op(self) -> Optional["IndividualOp"]:
        return None

    @property
    def _final_estimator(self) -> Any:
        op: Optional[IndividualOp] = self._final_individual_op
        model = None
        if op is not None:
            model = op.impl
        return "passthrough" if model is None else model

    @property
    def classes_(self):
        return self._final_estimator.classes_

    @property
    def n_classes_(self):
        return self._final_estimator.n_classes_

    @property
    def _get_tags(self):
        return self._final_estimator._get_tags

    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

    def get_param_ranges(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns two dictionaries, ranges and cat_idx, for hyperparameters.

        The ranges dictionary has two kinds of entries. Entries for
        numeric and Boolean hyperparameters are tuples of the form
        (min, max, default). Entries for categorical hyperparameters
        are lists of their values.

        The cat_idx dictionary has (min, max, default) entries of indices
        into the corresponding list of values.

        Warning: ignores side constraints and unions."""
        op: Optional[IndividualOp] = self._final_individual_op
        if op is None:
            raise ValueError("This pipeline does not end with an individual operator")
        return op.get_param_ranges()

    def get_param_dist(self, size=10) -> Dict[str, List[Any]]:
        """Returns a dictionary for discretized hyperparameters.

        Each entry is a list of values. For continuous hyperparameters,
        it returns up to `size` uniformly distributed values.

        Warning: ignores side constraints, unions, and distributions."""
        op: Optional[IndividualOp] = self._final_individual_op
        if op is None:
            raise ValueError("This pipeline does not end with an individual operator")
        return op.get_param_dist(size=size)

    # should this be abstract?  what do we do for grammars?
    def get_defaults(self) -> Mapping[str, Any]:
        return {}

    def clone(self) -> "Operator":
        """Return a copy of this operator, with the same hyper-parameters but without training data
        This behaves the same as calling sklearn.base.clone(self)
        """
        cp = clone(self)
        return cp

    def replace(
        self, original_op: "Operator", replacement_op: "Operator"
    ) -> "Operator":
        """Replaces an original operator with a replacement operator for the given operator.
        Replacement also occurs for all operators within the given operator's steps (i.e. pipelines and
        choices). If a planned operator is given as original_op, all derived operators (including
        trainable and trained versions) will be replaced. Otherwise, only the exact operator
        instance will be replaced.

        Parameters
        ----------
        original_op :
            Operator to replace within given operator. If operator is a planned operator,
            all derived operators (including trainable and trained versions) will be
            replaced. Otherwise, only the exact operator instance will be replaced.

        replacement_op :
            Operator to replace the original with.

        Returns
        -------
        modified_operator :
            Modified operator where original operator is replaced with replacement throughout.
        """

        def _check_match(subject, original_op):
            if (
                not isinstance(original_op, TrainableOperator)
                and isinstance(subject, IndividualOp)
                and isinstance(original_op, IndividualOp)
            ):
                # is planned operator, so replace any matching downstream operator
                if isinstance(subject, original_op):  # type: ignore
                    return True
            else:
                # is trainable or trained operator, only check exact instance match
                if subject == original_op:
                    return True
            return False

        @overload
        def _replace(
            subject: "Operator", original_op: "Operator", replacement_op: "Operator"
        ) -> "Operator": ...

        @overload
        def _replace(
            subject: list, original_op: "Operator", replacement_op: "Operator"
        ) -> list: ...

        @overload
        def _replace(
            subject: dict, original_op: "Operator", replacement_op: "Operator"
        ) -> dict: ...

        def _replace(subject, original_op: "Operator", replacement_op: "Operator"):
            # if operator has steps, recursively iterate through steps and recombine
            if hasattr(subject, "steps"):
                # special case if original_op has steps, check if it matches subject first
                if hasattr(original_op, "steps"):
                    if _check_match(subject, original_op):
                        return replacement_op

                new_steps: List[Operator] = []
                if isinstance(subject, BasePipeline):
                    # first convert pipeline edges to index-based representation
                    index_edges = []
                    for edge in subject.edges():
                        index_edges.append(
                            (
                                subject.steps_list().index(edge[0]),
                                subject.steps_list().index(edge[1]),
                            )
                        )

                    for step in subject.steps_list():
                        new_steps.append(_replace(step, original_op, replacement_op))

                    # use previous index-based representation to reconstruct edges
                    new_edges: List[Tuple[Operator, Operator]] = []
                    for index_tuple in index_edges:
                        new_edges.append(
                            (new_steps[index_tuple[0]], new_steps[index_tuple[1]])
                        )

                    return make_pipeline_graph(new_steps, new_edges)

                elif isinstance(subject, OperatorChoice):
                    for step in subject.steps_list():
                        new_steps.append(_replace(step, original_op, replacement_op))
                    return make_choice(*new_steps)

                else:
                    raise NotImplementedError(
                        "replace() needs to implement recombining this operator with steps"
                    )
            else:
                # base case for recursion: operator with no steps, returns replacement if applicable, original otherwise
                if _check_match(subject, original_op):
                    return replacement_op

                # special case of subject being in a collection
                if isinstance(subject, list):
                    return [_replace(s, original_op, replacement_op) for s in subject]
                elif isinstance(subject, tuple):
                    return tuple(
                        _replace(s, original_op, replacement_op) for s in subject
                    )
                elif isinstance(subject, dict):
                    return {
                        k: _replace(v, original_op, replacement_op)
                        for k, v in subject.items()
                    }

                # special case of hyperparams containing operators, usually referring to an estimator
                if hasattr(subject, "hyperparams") and subject.hyperparams():
                    modified_hyperparams = subject.hyperparams().copy()
                    for hyperparam, param_value in modified_hyperparams.items():
                        modified_hyperparams[hyperparam] = _replace(
                            param_value, original_op, replacement_op
                        )

                    return subject(**modified_hyperparams)

            return subject

        return _replace(self, original_op, replacement_op)  # type: ignore

    def with_params(self, **impl_params) -> "Operator":
        """This implements a functional version of set_params
        which returns a new operator instead of modifying the original
        """
        return self._with_params(False, **impl_params)

    @abstractmethod
    def _with_params(self, try_mutate: bool, **impl_params) -> "Operator":
        """
        This method updates the parameters of the operator.
        If try_mutate is set, it will attempt to update the operator in place,
        although this may not always be possible
        """
        pass

    def to_lale(self):
        """This is a deprecated method for backward compatibility and will be removed soon"""
        warnings.warn(
            "Operator.to_lale exists for backwards compatibility with make_sklearn_compat and will be removed soon",
            DeprecationWarning,
        )
        return self

    def __getattr__(self, name: str) -> Any:
        if name == "_cached_masked_attr_list":
            raise AttributeError()

        predict_methods = [
            "get_pipeline",
            "summary",
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
            "score",
            "score_samples",
            "predict_log_proba",
        ]
        if name in predict_methods:
            if isinstance(self, TrainedIndividualOp) or (
                isinstance(self, TrainableIndividualOp) and hasattr(self, "_trained")
            ):
                raise AttributeError(
                    f"The underlying operator implementation class does not define {name}"
                )
            if isinstance(self, TrainableIndividualOp) and not hasattr(
                self, "_trained"
            ):
                raise AttributeError(
                    f"{self.name()} is not trained. Note that in lale, the result of fit is a new trained operator that should be used with {name}."
                )
            if isinstance(self, PlannedOperator) and not isinstance(
                self, TrainableOperator
            ):
                pass  # as the plannedOperators are handled in a separate block next
            else:
                raise AttributeError(
                    f"Calling {name} on a {type(self)} is deprecated.  It needs to be trained by calling fit.  Note that in lale, the result of fit is a new TrainedOperator that should be used with {name}."
                )

        if name == "fit" or name in predict_methods:

            def get_error_msg(op, i):
                if isinstance(op, OperatorChoice):
                    error_msg = f"""[A.{i}] Please remove the operator choice `|` from `{op.name()}` and keep only one of those operators.\n"""
                elif isinstance(op, PlannedIndividualOp) and not isinstance(
                    op, TrainableIndividualOp
                ):
                    error_msg = f"[A.{i}] Please use `{op.name()}()` instead of `{op.name()}.`\n"
                else:
                    return ""
                return error_msg

            def add_error_msg_for_predict_methods(op, error_msg):
                if name in [
                    "get_pipeline",
                    "summary",
                    "transform",
                    "predict",
                    "predict_proba",
                    "decision_function",
                    "score",
                    "score_samples",
                    "predict_log_proba",
                ]:
                    error_msg = (
                        error_msg
                        + """\nAfter applying the suggested fixes the operator might need to be trained by calling fit ."""
                    )
                return error_msg

            # This method is called only when `name` is not found on the object, so
            # we don't need to account for the case when self is trainable or trained.
            if isinstance(self, PlannedIndividualOp):
                error_msg = f"""Please use `{self.name()}()` instead of `{self.name()}` to make it trainable.
Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the operator to use Hyperopt for
`max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`."""
                error_msg = add_error_msg_for_predict_methods(self, error_msg)
                raise AttributeError(error_msg)
            if isinstance(self, (PlannedPipeline, OperatorChoice)):
                error_msg = f"""The pipeline is not trainable, which means you can not call {name} on it.\n
Suggested fixes:\nFix [A]: You can make the following changes in the pipeline in order to make it trainable:\n"""
                i = 1
                if isinstance(self, PlannedPipeline):
                    for step in self.steps_list():
                        step_err = get_error_msg(step, i)
                        if step_err != "":
                            error_msg = error_msg + step_err
                            i += 1
                elif isinstance(self, OperatorChoice):
                    error_msg = error_msg + get_error_msg(self, i)

                error_msg = (
                    error_msg
                    + """\nFix [B]: Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the pipeline
to use Hyperopt for `max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`."""
                )
                error_msg = add_error_msg_for_predict_methods(self, error_msg)
                raise AttributeError(error_msg)

        forwards = self.get_forwards()
        if (
            forwards is True
            or (
                name.endswith("_")
                and not (name.startswith("__") and name.endswith("__"))
            )
            or (isinstance(forwards, list) and name in forwards)
        ):
            # we should try forwarding it.
            # first, a sanity check to prevent confusing behaviour where
            # forwarding works on a plannedoperator and then fails on a trainedoperator
            trained_ops = self._get_masked_attr_list()

            if name not in trained_ops:
                # ok, let us try to forward it
                # first we try the "shallow" wrapper,
                # and then we try each successive wrapped model
                model = self.shallow_impl
                while model is not None:
                    if hasattr(model, name):
                        return getattr(model, name)
                    old_model = model
                    model = getattr(model, "_wrapped_model", None)
                    if model is old_model:
                        model = None

        raise AttributeError(f"Attribute {name} not found for {self}")


Operator.__doc__ = cast(str, Operator.__doc__) + "\n" + _combinators_docstrings


class PlannedOperator(Operator):
    """Abstract class for Lale operators in the planned lifecycle state."""

    # pylint:disable=abstract-method

    def auto_configure(
        self,
        X: Any,
        y: Any = None,
        optimizer: "Optional[PlannedIndividualOp]" = None,
        cv: Any = None,
        scoring: Any = None,
        **kwargs,
    ) -> "TrainedOperator":
        """
        Perform combined algorithm selection and hyperparameter tuning on this planned operator.

        Parameters
        ----------
        X:
            Features that conform to the X property of input_schema_fit.
        y: optional
            Labels that conform to the y property of input_schema_fit.
            Default is None.
        optimizer:
            lale.lib.lale.Hyperopt or lale.lib.lale.GridSearchCV
            default is None.
        cv:
            cross-validation option that is valid for the optimizer.
            Default is None, which will use the optimizer's default value.
        scoring:
            scoring option that is valid for the optimizer.
            Default is None, which will use the optimizer's default value.
        kwargs:
            Other keyword arguments to be passed to the optimizer.

        Returns
        -------
        TrainableOperator
            Best operator discovered by the optimizer.

        Raises
        ------
        ValueError
            If an invalid optimizer is provided
        """
        if optimizer is None:
            raise ValueError("Please provide a valid optimizer for auto_configure.")
        if kwargs is None:
            kwargs = {}
        if cv is not None:
            kwargs["cv"] = cv
        if scoring is not None:
            kwargs["scoring"] = scoring
        optimizer_obj = optimizer(estimator=self, **kwargs)
        trained = optimizer_obj.fit(X, y)
        ret_pipeline = trained.get_pipeline()
        assert ret_pipeline is not None
        return ret_pipeline


PlannedOperator.__doc__ = (
    cast(str, PlannedOperator.__doc__) + "\n" + _combinators_docstrings
)


class TrainableOperator(PlannedOperator):
    """Abstract class for Lale operators in the trainable lifecycle state."""

    @overload
    def __and__(self, other: "TrainedOperator") -> "TrainablePipeline": ...

    @overload
    def __and__(self, other: "TrainableOperator") -> "TrainablePipeline": ...

    @overload
    def __and__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline": ...

    def __and__(self, other):  # type: ignore
        return make_union_no_concat(self, other)

    @overload
    def __rshift__(self, other: "TrainedOperator") -> "TrainablePipeline": ...

    @overload
    def __rshift__(self, other: "TrainableOperator") -> "TrainablePipeline": ...

    @overload
    def __rshift__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline": ...

    def __rshift__(self, other):  # type: ignore
        return make_pipeline(self, other)

    @abstractmethod
    def fit(self, X: Any, y: Any = None, **fit_params) -> "TrainedOperator":
        """Train the learnable coefficients of this operator, if any.

        Return a trained version of this operator.  If this operator
        has free learnable coefficients, bind them to values that fit
        the data according to the operator's algorithm.  Do nothing if
        the operator implementation lacks a `fit` method or if the
        operator has been marked as `is_frozen_trained`.

        Parameters
        ----------
        X:
            Features that conform to the X property of input_schema_fit.
        y: optional
            Labels that conform to the y property of input_schema_fit.
            Default is None.
        fit_params: Dictionary, optional
            A dictionary of keyword parameters to be used during training.

        Returns
        -------
        TrainedOperator
            A new copy of this operators that is the same except that its
            learnable coefficients are bound to their trained values.

        """
        pass

    def fit_transform(self, X: Any, y: Any = None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X:
            Features that conform to the X property of input_schema_fit.
        y: optional
            Labels that conform to the y property of input_schema_fit.
            Default is None.
        fit_params: Dictionary, optional
            A dictionary of keyword parameters to be used during training.

        Returns
        -------
        result :
            Transformed features; see output_transform schema of the operator.
        """
        return self.fit(X, y, **fit_params).transform(X)

    @abstractmethod
    def freeze_trainable(self) -> "TrainableOperator":
        """Return a copy of the trainable parts of this operator that is the same except
        that all hyperparameters are bound and none are free to be tuned.
        If there is an operator choice, it is kept as is.
        """
        pass

    @abstractmethod
    def is_transformer(self) -> bool:
        """Checks if the operator is a transformer"""
        pass


TrainableOperator.__doc__ = (
    cast(str, TrainableOperator.__doc__) + "\n" + _combinators_docstrings
)


class TrainedOperator(TrainableOperator):
    """Abstract class for Lale operators in the trained lifecycle state."""

    @overload
    def __and__(self, other: "TrainedOperator") -> "TrainedPipeline": ...

    @overload
    def __and__(self, other: "TrainableOperator") -> "TrainablePipeline": ...

    @overload
    def __and__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline": ...

    def __and__(self, other):  # type: ignore
        return make_union_no_concat(self, other)

    @overload
    def __rshift__(self, other: "TrainedOperator") -> "TrainedPipeline": ...

    @overload
    def __rshift__(self, other: "TrainableOperator") -> "TrainablePipeline": ...

    @overload
    def __rshift__(self, other: Union[Any, "Operator"]) -> "PlannedPipeline": ...

    def __rshift__(self, other):  # type: ignore
        return make_pipeline(self, other)

    @abstractmethod
    def transform(self, X: Any, y: Any = None) -> Any:
        """Transform the data.

        Parameters
        ----------
        X :
            Features; see input_transform schema of the operator.
        y : None

        Returns
        -------
        result :
            Transformed features; see output_transform schema of the operator.
        """
        pass

    @abstractmethod
    def _predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, X: Any, **predict_params) -> Any:
        """Make predictions.

        Parameters
        ----------
        X :
            Features; see input_predict schema of the operator.
        predict_params:
            Additional parameters that should be passed to the predict method

        Returns
        -------
        result :
            Predictions; see output_predict schema of the operator.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        pass

    @abstractmethod
    def decision_function(self, X: Any) -> Any:
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        pass

    @abstractmethod
    def score_samples(self, X: Any) -> Any:
        """Scores for each sample in X. The type of scores depends on the operator.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            scores per sample.
        """
        pass

    @abstractmethod
    def score(self, X: Any, y: Any, **score_params) -> Any:
        """Performance evaluation with a default metric.

        Parameters
        ----------
        X :
            Features.
        y:
            Ground truth labels.
        score_params:
            Any additional parameters expected by the score function of
            the underlying operator.
        Returns
        -------
        score :
            performance metric value
        """
        pass

    @abstractmethod
    def predict_log_proba(self, X: Any) -> Any:
        """Predicted class log-probabilities for X.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            Class log probabilities.
        """
        pass

    @abstractmethod
    def freeze_trained(self) -> "TrainedOperator":
        """Return a copy of this trainable operator that is the same except
        that all learnable coefficients are bound and thus fit is a no-op.
        """
        pass


TrainedOperator.__doc__ = (
    cast(str, TrainedOperator.__doc__) + "\n" + _combinators_docstrings
)

_schema_derived_attributes = ["_enum_attributes", "_hyperparam_defaults"]


class _DictionaryObjectForEnum:
    _d: Dict[str, enumeration.Enum]

    def __init__(self, d: Dict[str, enumeration.Enum]):
        self._d = d

    def __contains__(self, key: str) -> bool:
        return key in self._d

    # This method in fact always return an enumeration
    # however, the values of the enumeration are not known, which causes
    # the type checker to complain about a common (and desired) idiom
    # such as, e.g. LogisticRegression.enum.solver.saga
    # so we weaken the type to Any for pragmatic reasons
    def __getattr__(self, key: str) -> Any:  # enumeration.Enum:
        if key in self._d:
            return self._d[key]
        else:
            raise AttributeError("No enumeration found for hyper-parameter: " + key)

    # This method in fact always return an enumeration
    # however, the values of the enumeration are not known, which causes
    # the type checker to complain about a common (and desired) idiom
    # such as, e.g. LogisticRegression.enum.solver.saga
    # so we weaken the type to Any for pragmatic reasons
    def __getitem__(self, key: str) -> Any:  # enumeration.Enum:
        if key in self._d:
            return self._d[key]
        else:
            raise KeyError("No enumeration found for hyper-parameter: " + key)


class _WithoutGetParams:
    """This is a wrapper class whose job is to *NOT* have a get_params method,
    causing sklearn clone to call deepcopy on it (and its contents).
    This is currently used, for example, to wrap the impl class instance
    returned by an individual operator's get_params (since the class itself may have
    a get_params method defined, causing problems if this wrapper is not used).
    """

    @classmethod
    def unwrap(cls, obj):
        while isinstance(obj, _WithoutGetParams):
            obj = obj.klass
        return obj

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, _WithoutGetParams):
            return obj
        else:
            return _WithoutGetParams(obj)

    klass: type

    def __init__(self, klass: type):
        self.klass = klass


class IndividualOp(Operator):
    """
    This is a concrete class that can instantiate a new individual
    operator and provide access to its metadata.
    The enum property can be used to access enumerations for hyper-parameters,
    auto-generated from the operator's schema.
    For example, `LinearRegression.enum.solver.saga`
    As a short-hand, if the hyper-parameter name does not conflict with
    any fields of this class, the auto-generated enums can also be accessed
    directly.
    For example, `LinearRegression.solver.saga`"""

    _impl: Any
    _impl_class_: Union[type, _WithoutGetParams]
    _hyperparams: Optional[Dict[str, Any]]
    _frozen_hyperparams: Optional[List[str]]

    # this attribute may not be defined
    _hyperparam_defaults: Mapping[str, Any]

    def __init__(
        self,
        _lale_name: str,
        _lale_impl,
        _lale_schemas,
        _lale_frozen_hyperparameters=None,
        **hp,
    ) -> None:
        """Create a new IndividualOp.

        Parameters
        ----------
        name : String
            Name of the operator.
        impl :
            An instance of operator implementation class. This is a class that
            contains fit, predict/transform methods implementing an underlying
            algorithm.
        schemas : dict
            This is a dictionary of json schemas for the operator.
        """
        self._name = _lale_name
        self._enum_attributes = None
        if _lale_schemas:
            self._schemas = _lale_schemas
        else:
            self._schemas = get_default_schema(_lale_impl)

        # if we are given a class instance, we need to preserve it
        # so that get_params can return the same exact one that we got
        # this is important for scikit-learn's clone to work correctly
        unwrapped: Any = _WithoutGetParams.unwrap(_lale_impl)
        self._impl = unwrapped
        if inspect.isclass(unwrapped):
            self._impl_class_ = _lale_impl
        else:
            self._impl_class_ = unwrapped.__class__

        self._frozen_hyperparams = _lale_frozen_hyperparameters
        self._hyperparams = hp

    def _is_instantiated(self):
        return not inspect.isclass(self._impl)

    def _get_masked_attr_list(self):
        prev_cached_value = getattr(self, "_cached_masked_attr_list", None)
        if prev_cached_value is not None:
            return prev_cached_value
        found_ops = [
            "get_pipeline",
            "summary",
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
            "score",
            "score_samples",
            "predict_log_proba",
            "_schemas",
            "_impl",
            "_impl_class",
            "_hyperparams",
            "_frozen_hyperparams",
            "_trained",
            "_enum_attributes",
            "_cached_masked_attr_list",
        ]
        found_ops.extend(dir(TrainedIndividualOp))
        found_ops.extend(dir(self))
        self._cached_masked_attr_list = found_ops
        return found_ops

    def _check_schemas(self):
        from lale.settings import disable_hyperparams_schema_validation

        if disable_hyperparams_schema_validation:
            return

        validate_is_schema(self._schemas)
        from lale.pretty_print import json_to_string

        assert (
            self.has_tag("transformer") == self.is_transformer()
        ), f"{self.class_name()}: {json_to_string(self._schemas)}"
        assert self.has_tag("estimator") == self.has_method(
            "predict"
        ), f"{self.class_name()}: {json_to_string(self._schemas)}"
        if self.has_tag("classifier") or self.has_tag("regressor"):
            assert self.has_tag(
                "estimator"
            ), f"{self.class_name()}: {json_to_string(self._schemas)}"

        forwards = self.get_forwards()
        # if it is a boolean, there is nothing to check
        if isinstance(forwards, list):
            trained_ops = self._get_masked_attr_list()
            for f in forwards:
                assert (
                    f not in trained_ops
                ), f"""This operator specified the {f} attribute to be forwarded.
                Unfortunately, this method is also provided for some lale operator wrapper classes, so this
                is invalid.
                It is possible that this method/property is new to lale, and an older version of lale supported
                forwarding this method/property, however, to be compatible with this version of lale, the attribute needs
                to be removed from the forwards list, and code that calls this method/property (on an object op)
                need to be changed from op.{f} to op.impl.{f}
                """

        # Add enums from the hyperparameter schema to the object as fields
        # so that their usage looks like LogisticRegression.penalty.l1

    #        enum_gen.addSchemaEnumsAsFields(self, self.hyperparam_schema())

    _enum_attributes: Optional[_DictionaryObjectForEnum]

    @classmethod
    def _add_nested_params(cls, output: Dict[str, Any], k: str, v: Any):
        nested_params = cls._get_nested_params(v)
        if nested_params:
            output.update(nest_HPparams(k, nested_params))

    @classmethod
    def _get_nested_params(cls, v: Any) -> Optional[Dict[str, Any]]:
        # TODO: design question.  This seems like the right thing,
        # but sklearn does not currently do this, as is apparent with,
        # e.g VotingClassifier

        # if isinstance(v, list) or isinstance(v, tuple):
        #     output: Dict[str, Any] = {}
        #     for i, elem in enumerate(v):
        #         nested = cls._get_nested_params(elem)
        #         if nested:
        #             output.update(nest_HPparams(str(i)), nested)
        #     return output
        # elif isinstance(v, dict):
        #     output: Dict[str, Any] = {}
        #     for sub_k, sub_v in v.items():
        #         nested = cls._get_nested_params(sub_v)
        #         if nested:
        #             output.update(nest_HPparams(sub_k), nested)
        #     return output
        # else:
        try:
            return v.get_params(deep=True)
        except AttributeError:
            return None

    def _get_params_all(self, deep: bool = False) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        hps = self.hyperparams_all()
        if hps is not None:
            output.update(hps)
        defaults = self.get_defaults()
        for k in defaults.keys():
            if k not in output:
                output[k] = defaults[k]
        if deep:
            deep_stuff: Dict[str, Any] = {}
            for k, v in output.items():
                self._add_nested_params(deep_stuff, k, v)

            output.update(deep_stuff)
        return output

    def get_params(self, deep: Union[bool, Literal[0]] = True) -> Dict[str, Any]:
        """Get parameters for this operator.

        This method follows scikit-learn's convention that all operators
        have a constructor which takes a list of keyword arguments.
        This is not required for operator impls which do not desire
        scikit-compatibility.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this operator, and their nested parameters
            If False, will return the parameters for this operator, along with '_lale_XXX` fields needed to support cloning

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out: Dict[str, Any] = {}
        if deep is False:
            out["_lale_name"] = self._name
            out["_lale_schemas"] = self._schemas
            out["_lale_impl"] = _WithoutGetParams.wrap(self._wrapped_impl_class())
        # we need to stringify the class object, since the class object
        # has a get_params method (the instance method), which causes problems for
        # sklearn clone

        if self._is_instantiated():
            impl = self._impl_instance()
            if hasattr(impl, "get_params"):
                out.update(impl.get_params(deep=deep))
            elif hasattr(impl, "_wrapped_model") and hasattr(
                impl._wrapped_model, "get_params"
            ):
                out.update(impl._wrapped_model.get_params(deep=bool(deep)))
            else:
                out.update(self._get_params_all(deep=bool(deep)))
        else:
            out.update(self._get_params_all(deep=bool(deep)))

        if deep is False and self.frozen_hyperparams() is not None:
            out["_lale_frozen_hyperparameters"] = self.frozen_hyperparams()

        return out

    def _with_params(self, try_mutate: bool, **impl_params) -> "IndividualOp":
        main_params, partitioned_sub_params = partition_sklearn_params(impl_params)
        hyper = self.hyperparams()
        # we set the sub params first
        for sub_key, sub_params in partitioned_sub_params.items():
            with_structured_params(try_mutate, sub_key, sub_params, hyper)

        # we have now updated any nested operators
        # (if this is a higher order operator)
        # and can work on the main operator
        all_params = {**hyper, **main_params}
        filtered_impl_params = _fixup_hyperparams_dict(all_params)
        # These are used by lale.  Since they are returned by get_param
        # they may show up here (if the user calls get_param, changes
        # a values, and then calls set_param), so we remove them here

        filtered_impl_params.pop("_lale_name", None)
        filtered_impl_params.pop("_lale_impl", None)
        filtered_impl_params.pop("_lale_schemas", None)
        filtered_impl_params.pop("_lale_frozen_hyperparameters", None)

        return self._with_op_params(try_mutate, **filtered_impl_params)

    def _with_op_params(
        self, try_mutate: bool, **impl_params
    ) -> "TrainableIndividualOp":
        # for an individual (and planned individual) operator,
        # we don't mutate the operator itself even if try_mutate is True

        res = self._configure(**impl_params)
        return res

    # we have different views on the hyperparameters
    def hyperparams_all(self) -> Optional[Dict[str, Any]]:
        """This is the hyperparameters that are currently set.
        Some of them may not have been set explicitly
        (e.g. if this is a clone of an operator,
        some of these may be defaults.
        To get the hyperparameters that were actually set,
        use :meth:`hyperparams`
        """
        return getattr(self, "_hyperparams", None)

    def frozen_hyperparams(self) -> Optional[List[str]]:
        return self._frozen_hyperparams

    def _hyperparams_helper(self) -> Optional[Dict[str, Any]]:
        actuals = self.hyperparams_all()
        if actuals is None:
            return None
        frozen_params = self.frozen_hyperparams()
        if frozen_params is None:
            return None
        params = {k: actuals[k] for k in frozen_params}
        return params

    def hyperparams(self) -> Dict[str, Any]:
        params = self._hyperparams_helper()
        if params is None:
            return {}
        else:
            return params

    def reduced_hyperparams(self):
        actuals = self._hyperparams_helper()
        if actuals is None:
            return None
        defaults = self.get_defaults()
        actuals_minus_defaults = {
            k: actuals[k]
            for k in actuals
            if k not in defaults
            or not are_hyperparameters_equal(actuals[k], defaults[k])
        }

        if not hasattr(self, "_hyperparam_positionals"):
            sig = inspect.signature(self._impl_class().__init__)
            positionals = {
                name: defaults[name]
                for name, param in sig.parameters.items()
                if name != "self"
                and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and param.default == inspect.Parameter.empty
            }
            self._hyperparam_positionals = positionals
        result = {**self._hyperparam_positionals, **actuals_minus_defaults}
        return result

    def _configure(self, *args, **kwargs) -> "TrainableIndividualOp":
        class_ = self._impl_class()
        hyperparams = {}
        for arg in args:
            k, v = self._enum_to_strings(arg)
            hyperparams[k] = v
        for k, v in _fixup_hyperparams_dict(kwargs).items():
            if k in hyperparams:
                raise ValueError(f"Duplicate argument {k}.")
            v = val_wrapper.unwrap(v)
            if isinstance(v, enumeration.Enum):
                k2, v2 = self._enum_to_strings(v)
                if k != k2:
                    raise ValueError(f"Invalid keyword {k2} for argument {v2}.")
            else:
                v2 = v
            hyperparams[k] = v2
        frozen_hyperparams = list(hyperparams.keys())
        # using params_all instead of hyperparams to ensure the construction is consistent with schema
        trainable_to_get_params = TrainableIndividualOp(
            _lale_name=self.name(),
            _lale_impl=class_,
            _lale_schemas=self._schemas,
            _lale_frozen_hyperparameters=frozen_hyperparams,
            **hyperparams,
        )
        # TODO: improve this code
        params_all = trainable_to_get_params._get_params_all()
        self._validate_hyperparams(
            hyperparams, params_all, self.hyperparam_schema(), class_
        )
        # TODO: delay creating the impl here
        if len(params_all) == 0:
            impl = class_()
        else:
            impl = class_(**params_all)

        if self._should_configure_trained(impl):
            result: TrainableIndividualOp = TrainedIndividualOp(
                _lale_name=self.name(),
                _lale_impl=impl,
                _lale_schemas=self._schemas,
                _lale_frozen_hyperparameters=frozen_hyperparams,
                _lale_trained=True,
                **hyperparams,
            )
        else:
            result = TrainableIndividualOp(
                _lale_name=self.name(),
                _lale_impl=impl,
                _lale_schemas=self._schemas,
                _lale_frozen_hyperparameters=frozen_hyperparams,
                **hyperparams,
            )
        return result

    @property
    def enum(self) -> _DictionaryObjectForEnum:
        ea = getattr(self, "_enum_attributes", None)
        if ea is None:
            nea = enum_gen.schemaToPythonEnums(self.hyperparam_schema())
            doe = _DictionaryObjectForEnum(nea)
            self._enum_attributes = doe
            return doe
        else:
            return ea

    def _invalidate_enum_attributes(self) -> None:
        for k in _schema_derived_attributes:
            try:
                delattr(self, k)
            except AttributeError:
                pass

    def __getattr__(self, name: str) -> Any:
        if name in _schema_derived_attributes or name in ["__setstate__", "_schemas"]:
            raise AttributeError
        if name == "_estimator_type":
            if self.is_classifier():
                return "classifier"  # satisfy sklearn.base.is_classifier(op)
            elif self.is_regressor():
                return "regressor"  # satisfy sklearn.base.is_regressor(op)

        return super().__getattr__(name)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove entries that can't be pickled
        for k in _schema_derived_attributes:
            state.pop(k, None)
        return state

    def get_schema(self, schema_kind: str) -> Dict[str, Any]:
        """Return a schema of the operator.

        Parameters
        ----------
        schema_kind : string, 'hyperparams' or 'input_fit' or 'input_partial_fit' or 'input_transform'  or 'input_transform_X_y' or 'input_predict' or 'input_predict_proba' or 'input_decision_function' or 'output_transform' or 'output_transform_X_y' or 'output_predict' or 'output_predict_proba' or 'output_decision_function'
                Type of the schema to be returned.

        Returns
        -------
        dict
            The Python object containing the JSON schema of the operator.
            For all the schemas currently present, this would be a dictionary.
        """
        props = self._schemas["properties"]
        assert (
            schema_kind in props
        ), f"missing schema {schema_kind} for operator {self.name()} with class {self.class_name()}"
        result = props[schema_kind]
        return result

    def has_schema(self, schema_kind: str) -> bool:
        """Return true if the operator has the schema kind.

        Parameters
        ----------
        schema_kind : string, 'hyperparams' or 'input_fit' or 'input_partial_fit' or 'input_transform'  or 'input_transform_X_y' or 'input_predict' or 'input_predict_proba' or 'input_decision_function' or 'output_transform' or 'output_transform_X_y' or 'output_predict' or 'output_predict_proba' or 'output_decision_function' or 'input_score_samples' or 'output_score_samples'
                Type of the schema to be returned.

        Returns
        -------
        True if the json schema is present, False otherwise.
        """
        props = self._schemas["properties"]
        return schema_kind in props

    def documentation_url(self):
        if "documentation_url" in self._schemas:
            return self._schemas["documentation_url"]
        return None

    def get_forwards(self) -> Union[bool, List[str]]:
        """Returns the list of attributes (methods/properties)
        the schema has asked to be forwarded.  A boolean value is a blanket
        opt-in or out of forwarding
        """
        forwards = self._schemas.get("forwards", False)
        assert isinstance(
            forwards, (bool, list)
        ), f"the schema forward declaration {forwards} must be either a boolean or a list of strings"
        return forwards

    def get_tags(self) -> Dict[str, List[str]]:
        """Return the tags of an operator.

        Returns
        -------
        list
            A list of tags describing the operator.
        """
        return self._schemas.get("tags", {})

    def has_tag(self, tag: str) -> bool:
        """Check the presence of a tag for an operator.

        Parameters
        ----------
        tag : string

        Returns
        -------
        boolean
            Flag indicating the presence or absence of the given tag
            in this operator's schemas.
        """
        tags = [t for ll in self.get_tags().values() for t in ll]
        return tag in tags

    def input_schema_fit(self) -> JSON_TYPE:
        """Input schema for the fit method."""
        return self.get_schema("input_fit")

    def input_schema_partial_fit(self) -> JSON_TYPE:
        """Input schema for the partial_fit method."""
        return self.get_schema("input_partial_fit")

    def input_schema_transform(self) -> JSON_TYPE:
        """Input schema for the transform method."""
        return self.get_schema("input_transform")

    def input_schema_transform_X_y(self) -> JSON_TYPE:
        """Input schema for the transform_X_y method."""
        return self.get_schema("input_transform_X_y")

    def input_schema_predict(self) -> JSON_TYPE:
        """Input schema for the predict method."""
        return self.get_schema("input_predict")

    def input_schema_predict_proba(self) -> JSON_TYPE:
        """Input schema for the predict_proba method."""
        return self.get_schema("input_predict_proba")

    def input_schema_predict_log_proba(self) -> JSON_TYPE:
        """Input schema for the predict_log_proba method.
        We assume that it is the same as the predict_proba method if none has been defined explicitly.
        """
        if self.has_schema("input_predict_log_proba"):
            return self.get_schema("input_predict_log_proba")
        else:
            return self.get_schema("input_predict_proba")

    def input_schema_decision_function(self) -> JSON_TYPE:
        """Input schema for the decision_function method."""
        return self.get_schema("input_decision_function")

    def input_schema_score_samples(self) -> JSON_TYPE:
        """Input schema for the score_samples method.
        We assume that it is the same as the predict method if none has been defined explicitly.
        """
        if self.has_schema("input_score_samples"):
            return self.get_schema("input_score_samples")
        else:
            return self.get_schema("input_predict")

    def output_schema_transform(self) -> JSON_TYPE:
        """Oputput schema for the transform method."""
        return self.get_schema("output_transform")

    def output_schema_transform_X_y(self) -> JSON_TYPE:
        """Oputput schema for the transform_X_y method."""
        return self.get_schema("output_transform_X_y")

    def output_schema_predict(self) -> JSON_TYPE:
        """Output schema for the predict method."""
        return self.get_schema("output_predict")

    def output_schema_predict_proba(self) -> JSON_TYPE:
        """Output schema for the predict_proba method."""
        return self.get_schema("output_predict_proba")

    def output_schema_decision_function(self) -> JSON_TYPE:
        """Output schema for the decision_function method."""
        return self.get_schema("output_decision_function")

    def output_schema_score_samples(self) -> JSON_TYPE:
        """Output schema for the score_samples method.
        We assume that it is the same as the predict method if none has been defined explicitly.
        """
        if self.has_schema("output_score_samples"):
            return self.get_schema("output_score_samples")
        else:
            return self.get_schema("output_predict")

    def output_schema_predict_log_proba(self) -> JSON_TYPE:
        """Output schema for the predict_log_proba method.
        We assume that it is the same as the predict_proba method if none has been defined explicitly.
        """
        if self.has_schema("output_predict_log_proba"):
            return self.get_schema("output_predict_log_proba")
        else:
            return self.get_schema("output_predict_proba")

    def hyperparam_schema(self, name: Optional[str] = None) -> JSON_TYPE:
        """Returns the hyperparameter schema for the operator.

        Parameters
        ----------
        name : string, optional
            Name of the hyperparameter.

        Returns
        -------
        dict
            Full hyperparameter schema for this operator or part of the schema
            corresponding to the hyperparameter given by parameter `name`.
        """
        hp_schema = self.get_schema("hyperparams")
        if name is None:
            return hp_schema
        else:
            params = next(iter(hp_schema.get("allOf", [])))
            return params.get("properties", {}).get(name)

    def get_defaults(self) -> Mapping[str, Any]:
        """Returns the default values of hyperparameters for the operator.

        Returns
        -------
        dict
            A dictionary with names of the hyperparamers as keys and
            their default values as values.
        """
        if not hasattr(self, "_hyperparam_defaults"):
            schema = self.hyperparam_schema()
            props_container: Dict[str, Any] = next(iter(schema.get("allOf", [])), {})
            props: Dict[str, Any] = props_container.get("properties", {})
            # since we want to share this, we don't want callers
            # to modify the returned dictionary, htereby modifying the defaults
            defaults: MappingProxyType[str, Any] = MappingProxyType(
                {k: props[k].get("default") for k in props.keys()}
            )
            self._hyperparam_defaults = defaults
        return self._hyperparam_defaults

    def get_param_ranges(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns two dictionaries, ranges and cat_idx, for hyperparameters.

        The ranges dictionary has two kinds of entries. Entries for
        numeric and Boolean hyperparameters are tuples of the form
        (min, max, default). Entries for categorical hyperparameters
        are lists of their values.

        The cat_idx dictionary has (min, max, default) entries of indices
        into the corresponding list of values.

        Warning: ignores side constraints and unions."""

        hyperparam_obj = next(iter(self.hyperparam_schema().get("allOf", [])))
        original = hyperparam_obj.get("properties")

        def is_for_optimizer(s) -> bool:
            return ("forOptimizer" not in s) or s["forOptimizer"]

        def is_relevant(hp, s):
            if "relevantToOptimizer" in hyperparam_obj:
                return hp in hyperparam_obj["relevantToOptimizer"]
            return True

        relevant = {hp: s for hp, s in original.items() if is_relevant(hp, s)}

        def pick_one_type(schema):
            if not is_for_optimizer(schema):
                return None
            if "anyOf" in schema:

                def by_type(typ):
                    for s in schema["anyOf"]:
                        if "type" in s and s["type"] == typ:
                            if is_for_optimizer(s):
                                return s
                    return None

                s = None
                for typ in ["number", "integer", "string"]:
                    s = by_type(typ)
                    if s:
                        return s
                if s is None:
                    enums = []
                    for s in schema["anyOf"]:
                        if "enum" in s:
                            if is_for_optimizer(s):
                                enums.append(s)
                        elif s.get("type", None) == "boolean":
                            if is_for_optimizer(s):
                                bool_s = {"enum": [False, True]}
                                d = s.get("default", None)
                                if d is not None:
                                    bool_s["default"] = d

                                enums.append(bool_s)
                    if len(enums) == 1:
                        return enums[0]
                    elif enums:
                        # combine them, and see if there is an anyOf default that we want to use as well
                        vals = [item for s in enums for item in s["enum"]]
                        new_s = {
                            "enum": vals,
                        }

                        if "default" in schema and schema["default"] in vals:
                            new_s["default"] = schema["default"]
                        else:
                            for s in enums:
                                if "default" in s:
                                    new_s["default"] = s["default"]
                                    break
                        return new_s
                if len(schema["anyOf"]) > 0 and is_for_optimizer(schema["anyOf"][0]):
                    return schema["anyOf"][0]
                else:
                    return None
            return schema

        unityped_with_none = {hp: pick_one_type(relevant[hp]) for hp in relevant}

        unityped = {k: v for k, v in unityped_with_none.items() if v is not None}

        def add_default(schema):
            if "type" in schema:
                minimum, maximum = 0.0, 1.0
                if "minimumForOptimizer" in schema:
                    minimum = schema["minimumForOptimizer"]
                elif "minimum" in schema:
                    minimum = schema["minimum"]
                if "maximumForOptimizer" in schema:
                    maximum = schema["maximumForOptimizer"]
                elif "maximum" in schema:
                    maximum = schema["maximum"]
                result = {**schema}
                if schema["type"] in ["number", "integer"]:
                    if "default" not in schema:
                        schema["default"] = None
                    if "minimumForOptimizer" not in schema:
                        result["minimumForOptimizer"] = minimum
                    if "maximumForOptimizer" not in schema:
                        result["maximumForOptimizer"] = maximum
                return result
            elif "enum" in schema:
                if "default" in schema:
                    return schema
                return {"default": schema["enum"][0], **schema}
            return schema

        defaulted = {hp: add_default(unityped[hp]) for hp in unityped}

        def get_range(hp, schema):
            if "enum" in schema:
                default = schema["default"]
                non_default = [v for v in schema["enum"] if v != default]
                return [*non_default, default]
            elif schema["type"] == "boolean":
                return (False, True, schema["default"])
            else:

                def get(schema, key):
                    return schema[key] if key in schema else None

                keys = ["minimumForOptimizer", "maximumForOptimizer", "default"]
                return tuple(get(schema, key) for key in keys)

        def get_cat_idx(schema):
            if "enum" not in schema:
                return None
            return (0, len(schema["enum"]) - 1, len(schema["enum"]) - 1)

        autoai_ranges = {hp: get_range(hp, s) for hp, s in defaulted.items()}
        if "min_samples_split" in autoai_ranges and "min_samples_leaf" in autoai_ranges:
            if self._name not in (
                "_GradientBoostingRegressorImpl",
                "_GradientBoostingClassifierImpl",
                "_ExtraTreesClassifierImpl",
            ):
                autoai_ranges["min_samples_leaf"] = (1, 5, 1)
                autoai_ranges["min_samples_split"] = (2, 5, 2)
        autoai_cat_idx = {
            hp: get_cat_idx(s) for hp, s in defaulted.items() if "enum" in s
        }
        return autoai_ranges, autoai_cat_idx

    def get_param_dist(self, size=10) -> Dict[str, List[Any]]:
        """Returns a dictionary for discretized hyperparameters.

        Each entry is a list of values. For continuous hyperparameters,
        it returns up to `size` uniformly distributed values.

        Warning: ignores side constraints, unions, and distributions."""
        autoai_ranges, _autoai_cat_idx = self.get_param_ranges()

        def one_dist(key: str) -> List[Any]:
            one_range = autoai_ranges[key]
            if isinstance(one_range, tuple):
                minimum, maximum, default = one_range
                if minimum is None:
                    dist = [default]
                elif isinstance(minimum, bool):
                    if minimum == maximum:
                        dist = [minimum]
                    else:
                        dist = [minimum, maximum]
                elif isinstance(minimum, int) and isinstance(maximum, int):
                    step = float(maximum - minimum) / (size - 1)
                    fdist = [minimum + i * step for i in range(size)]
                    dist = list(set(round(f) for f in fdist))
                    dist.sort()
                elif isinstance(minimum, (int, float)):
                    # just in case the minimum or maximum is exclusive
                    epsilon = (maximum - minimum) / (100 * size)
                    minimum += epsilon
                    maximum -= epsilon
                    step = (maximum - minimum) / (size - 1)
                    dist = [minimum + i * step for i in range(size)]
                else:
                    assert False, f"key {key}, one_range {one_range}"
            else:
                dist = [*one_range]
            return dist

        autoai_dists = {k: one_dist(k) for k in autoai_ranges.keys()}
        return autoai_dists

    def _enum_to_strings(self, arg: "enumeration.Enum") -> Tuple[str, Any]:
        """[summary]

        Parameters
        ----------
        arg : [type]
            [description]

        Raises
        ------
        ValueError
            [description]

        Returns
        -------
        [type]
            [description]
        """

        if not isinstance(arg, enumeration.Enum):
            raise ValueError(f"Missing keyword on argument {arg}.")
        return arg.__class__.__name__, arg.value

    def _wrapped_impl_class(self):
        if not hasattr(self, "_impl_class_"):
            if inspect.isclass(self._impl):
                self._impl_class_ = self._impl
            else:
                self._impl_class_ = self._impl.__class__
        return self._impl_class_

    def _impl_class(self):
        return _WithoutGetParams.unwrap(self._wrapped_impl_class())

    def _impl_instance(self) -> Any:
        hyperparams: Mapping[str, Any]

        if not self._is_instantiated():
            defaults = self.get_defaults()
            all_hps = self.hyperparams_all()
            if all_hps:
                hyperparams = {**defaults, **all_hps}
            else:
                hyperparams = defaults

            class_ = self._impl_class()
            try:
                instance = class_(
                    **hyperparams
                )  # always with default values of hyperparams
            except TypeError as e:
                logger.debug(
                    f"Constructor for {class_.__module__}.{class_.__name__} "
                    f"threw exception {e}"
                )
                # TODO: Is this really a reasonable fallback?
                instance = class_.__new__()  # type:ignore
            self._impl = instance
        return self._impl

    @property
    def impl(self) -> Any:
        """Returns the underlying impl.  This can be used to access additional
        field and methods not exposed by Lale.  If only the type of the
        impl is needed, please use self.impl_class instead, as it can be more efficient.

        If the found impl has a _wrapped_model, it will be returned instead
        """
        model = self.shallow_impl
        if model is None:
            return None
        while True:
            base_model = getattr(model, "_wrapped_model", model)
            if base_model is None or base_model is model:
                return model
            model = base_model
        return model

    @property
    def shallow_impl(self) -> Any:
        """Returns the underlying impl.  This can be used to access additional
        field and methods not exposed by Lale.  If only the type of the
        impl is needed, please use self.impl_class instead, as it can be more efficient.
        """
        # if fit was called, we want to use trained result
        # even if the code uses the original operrator
        # since sklearn assumes that fit mutates the operator
        op = self
        if hasattr(op, "_trained"):
            tr_op: Any = op._trained
            if tr_op is not None:
                assert isinstance(tr_op, TrainedIndividualOp)
                op = tr_op
        return op._impl_instance()

    @property
    def impl_class(self) -> type:
        """Returns the class of the underlying impl. This should return the same thing
        as self.impl.__class__, but can be more efficient.
        """
        return self._impl_class()

    # This allows the user, for example, to check isinstance(LR().fit(...), LR)
    def __instancecheck__(self, other):
        if isinstance(other, IndividualOp):
            return issubclass(other.impl_class, self.impl_class)
        else:
            return False

    def class_name(self) -> str:
        module = None
        if self._impl is not None:
            module = self._impl.__module__
        if module is None or module == str.__class__.__module__:  # type: ignore
            class_name = self.name()
        else:
            class_name = module + "." + self._impl_class().__name__
        return class_name

    def __str__(self) -> str:
        return self.name()

    # # sklearn calls __repr__ instead of __str__
    def __repr__(self):
        name = self.name()
        return name

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator implementations are compatible"""
        if not isinstance(other, IndividualOp):
            return False
        return self._impl_class() == other._impl_class()

    def _propose_fixed_hyperparams(
        self, key_candidates, hp_all, hp_schema, max_depth=2
    ):
        defaults = self.get_defaults()
        explicit_defaults: Dict[str, Any] = {k: defaults[k] for k in key_candidates}

        found: bool = False

        for depth in range(0, max_depth):
            if found:
                return
            candidate_replacements: Any = list(
                itertools.combinations(explicit_defaults.items(), depth + 1)
            )
            for replacements in candidate_replacements:
                new_values = dict(replacements)
                fixed_hp = {**hp_all, **new_values}
                try:
                    validate_schema_directly(fixed_hp, hp_schema)
                    found = True
                    yield new_values
                except jsonschema.ValidationError:
                    pass

    MAX_FIX_DEPTH: int = 2
    MAX_FIX_SUGGESTIONS: int = 3

    def _validate_hyperparams(self, hp_explicit, hp_all, hp_schema, class_):
        from lale.settings import disable_hyperparams_schema_validation

        if disable_hyperparams_schema_validation:
            return

        try:
            validate_schema_directly(hp_all, hp_schema)
        except jsonschema.ValidationError as e_orig:
            e = e_orig if e_orig.parent is None else e_orig.parent
            sch = e.schema
            assert isinstance(sch, dict)
            validate_is_schema(sch)
            schema = lale.pretty_print.to_string(sch)

            defaults = self.get_defaults()
            extra_keys = [k for k in hp_explicit.keys() if k not in defaults]
            trimmed_valid: bool = False
            if extra_keys:
                trimmed_hp_all = {
                    k: v for k, v in hp_all.items() if k not in extra_keys
                }
                trimmed_hp_explicit_keys = {
                    k for k in hp_explicit.keys() if k not in extra_keys
                }
                remove_recommendation = (
                    "unknown key"
                    + ("s" if len(extra_keys) > 1 else "")
                    + " "
                    + ", ".join(("'" + k + "'" for k in extra_keys))
                )

                try:
                    validate_schema_directly(trimmed_hp_all, hp_schema)
                    trimmed_valid = True
                except jsonschema.ValidationError:
                    pass

            else:
                trimmed_hp_all = hp_all
                trimmed_hp_explicit_keys = hp_explicit.keys()
                remove_recommendation = ""

            proposed_fix: str = ""
            if trimmed_valid and remove_recommendation:
                proposed_fix = "To fix, please remove " + remove_recommendation + "\n"
            else:
                find_fixed_hyperparam_iter = self._propose_fixed_hyperparams(
                    trimmed_hp_explicit_keys,
                    trimmed_hp_all,
                    hp_schema,
                    max_depth=self.MAX_FIX_DEPTH,
                )
                fix_suggestions: List[Dict[str, Any]] = list(
                    itertools.islice(
                        find_fixed_hyperparam_iter, self.MAX_FIX_SUGGESTIONS
                    )
                )
                if fix_suggestions:
                    from lale.pretty_print import hyperparams_to_string

                    if remove_recommendation:
                        remove_recommendation = (
                            "remove " + remove_recommendation + " and "
                        )
                    proposed_fix = "Some possible fixes include:\n" + "".join(
                        (
                            "- "
                            + remove_recommendation
                            + "set "
                            + hyperparams_to_string(d)
                            + "\n"
                            for d in fix_suggestions
                        )
                    )

            if [*e.schema_path][:3] == ["allOf", 0, "properties"]:
                arg = e.schema_path[3]
                reason = f"invalid value {arg}={e.instance}"
                schema_path = f"argument {arg}"
            elif [*e.schema_path][:3] == ["allOf", 0, "additionalProperties"]:
                pref, suff = "Additional properties are not allowed (", ")"
                assert e.message.startswith(pref) and e.message.endswith(suff)
                reason = "argument " + e.message[len(pref) : -len(suff)]
                schema_path = "arguments and their defaults"
                schema = self.get_defaults()
            elif e.schema_path[0] == "allOf" and int(e.schema_path[1]) != 0:
                assert e.schema_path[2] == "anyOf"
                schema = e.schema
                if isinstance(schema, dict):
                    descr = schema["description"]
                else:
                    descr = "Boolean schema"

                if descr.endswith("."):
                    descr = descr[:-1]
                reason = f"constraint {descr[0].lower()}{descr[1:]}"
                schema_path = "failing constraint"
                if self.documentation_url() is not None:
                    schema = f"{self.documentation_url()}#constraint-{e.schema_path[1]}"
            else:
                reason = e.message
                schema_path = e.schema_path
            msg = (
                f"Invalid configuration for {self.name()}("
                + f"{lale.pretty_print.hyperparams_to_string(hp_explicit if hp_explicit else {})}) "
                + f"due to {reason}.\n"
                + proposed_fix
                + f"Schema of {schema_path}: {schema}\n"
                + f"Invalid value: {e.instance}"
            )
            raise jsonschema.ValidationError(msg)
        user_validator = getattr(class_, "validate_hyperparams", None)
        if user_validator:
            user_validator(**hp_all)

    def validate_schema(self, X: Any, y: Any = None):
        if self.has_method("fit"):
            X = self._validate_input_schema("X", X, "fit")
        method = "transform" if self.is_transformer() else "predict"
        self._validate_input_schema("X", X, method)
        if self.is_supervised(default_if_missing=False):
            if y is None:
                raise ValueError(f"{self.name()}.fit() y cannot be None")
            if self.has_method("fit"):
                y = self._validate_input_schema("y", y, "fit")
            self._validate_input_schema("y", y, method)

    def _validate_input_schema(self, arg_name: str, arg, method: str):
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return arg

        if not is_empty_dict(arg):
            if method == "fit":
                schema = self.input_schema_fit()
            elif method == "partial_fit":
                schema = self.input_schema_partial_fit()
            elif method == "transform":
                schema = self.input_schema_transform()
            elif method == "transform_X_y":
                schema = self.input_schema_transform_X_y()
            elif method == "predict":
                schema = self.input_schema_predict()
            elif method == "predict_proba":
                schema = self.input_schema_predict_proba()
            elif method == "predict_log_proba":
                schema = self.input_schema_predict_log_proba()
            elif method == "decision_function":
                schema = self.input_schema_decision_function()
            elif method == "score_samples":
                schema = self.input_schema_score_samples()
            else:
                raise ValueError(f"Unexpected method argument: {method}")
            if "properties" in schema and arg_name in schema["properties"]:
                arg = add_schema(arg)
                try:
                    sup: JSON_TYPE = schema["properties"][arg_name]
                    validate_schema(arg, sup)
                except SubschemaError as e:
                    sub_str: str = lale.pretty_print.json_to_string(e.sub)
                    sup_str: str = lale.pretty_print.json_to_string(e.sup)
                    raise ValueError(
                        f"{self.name()}.{method}() invalid {arg_name}, the schema of the actual data is not a subschema of the expected schema of the argument.\nactual_schema = {sub_str}\nexpected_schema = {sup_str}"
                    ) from None
                except Exception as e:
                    exception_type = f"{type(e).__module__}.{type(e).__name__}"
                    raise ValueError(
                        f"{self.name()}.{method}() invalid {arg_name}: {exception_type}: {e}"
                    ) from None
        return arg

    def _validate_output_schema(self, result, method):
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return result

        if method == "transform":
            schema = self.output_schema_transform()
        elif method == "transform_X_y":
            schema = self.output_schema_transform_X_y()
        elif method == "predict":
            schema = self.output_schema_predict()
        elif method == "predict_proba":
            schema = self.output_schema_predict_proba()
        elif method == "predict_log_proba":
            schema = self.output_schema_predict_log_proba()
        elif method == "decision_function":
            schema = self.output_schema_decision_function()
        elif method == "score_samples":
            schema = self.output_schema_score_samples()
        else:
            raise ValueError(f"Unexpected method argument: {method}")

        result = add_schema(result)
        try:
            validate_schema(result, schema)
        except Exception as e:
            print(f"{self.name()}.{method}() invalid result: {e}")
            raise ValueError(f"{self.name()}.{method}() invalid result: {e}") from e
        return result

    def transform_schema(self, s_X: JSON_TYPE) -> JSON_TYPE:
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return {}
        elif self.is_transformer():
            return self.output_schema_transform()
        elif self.has_method("predict_proba"):
            return self.output_schema_predict_proba()
        elif self.has_method("decision_function"):
            return self.output_schema_decision_function()
        else:
            return self.output_schema_predict()

    def is_supervised(self, default_if_missing=True) -> bool:
        if self.has_method("fit"):
            schema_fit = self.input_schema_fit()
            # first we try a fast path, since subschema checking can be a bit slow
            if (
                schema_fit is not None
                and isinstance(schema_fit, dict)
                and all(
                    k not in schema_fit for k in ["all_of", "any_of", "one_of", "not"]
                )
            ):
                req = schema_fit.get("required", None)
                return req is not None and "y" in req
            else:
                return is_subschema(schema_fit, _is_supervised_schema)
        return default_if_missing

    def is_classifier(self) -> bool:
        return self.has_tag("classifier")

    def is_regressor(self) -> bool:
        return self.has_tag("regressor")

    def has_method(self, method_name: str) -> bool:
        return hasattr(self._impl, method_name)

    def is_transformer(self) -> bool:
        """Checks if the operator is a transformer"""
        return self.has_method("transform")

    @property
    def _final_individual_op(self) -> Optional["IndividualOp"]:
        return self


_is_supervised_schema = {"type": "object", "required": ["y"]}


class PlannedIndividualOp(IndividualOp, PlannedOperator):
    """
    This is a concrete class that returns a trainable individual
    operator through its __call__ method. A configure method can use
    an optimizer and return the best hyperparameter combination.
    """

    _hyperparams: Optional[Dict[str, Any]]

    def __init__(
        self,
        _lale_name: str,
        _lale_impl,
        _lale_schemas,
        _lale_frozen_hyperparameters=None,
        _lale_trained=False,
        **hp,
    ) -> None:
        super().__init__(
            _lale_name, _lale_impl, _lale_schemas, _lale_frozen_hyperparameters, **hp
        )

    def _should_configure_trained(self, impl):
        # TODO: may also want to do this for other higher-order operators
        if self.class_name() == _LALE_SKL_PIPELINE:
            return isinstance(impl._pipeline, TrainedPipeline)
        else:
            return not hasattr(impl, "fit")

    # give it a more precise type: if the input is an individual op, the output is as well
    def auto_configure(
        self, X: Any, y: Any = None, optimizer=None, cv=None, scoring=None, **kwargs
    ) -> "TrainedIndividualOp":
        trained = super().auto_configure(
            X, y=y, optimizer=optimizer, cv=cv, scoring=scoring, **kwargs
        )
        assert isinstance(trained, TrainedIndividualOp)
        return trained

    def __call__(self, *args, **kwargs) -> "TrainableIndividualOp":
        return self._configure(*args, **kwargs)

    def _hyperparam_schema_with_hyperparams(
        self, data_schema: Optional[Dict[str, Any]] = None
    ):
        def fix_hyperparams(schema):
            hyperparams = self.hyperparams()
            if not hyperparams:
                return schema
            props = {k: {"enum": [v]} for k, v in hyperparams.items()}
            obj = {"type": "object", "properties": props}
            obj["relevantToOptimizer"] = list(hyperparams.keys())
            obj["required"] = list(hyperparams.keys())
            top = {"allOf": [schema, obj]}
            return top

        s_1 = self.hyperparam_schema()
        s_2 = fix_hyperparams(s_1)
        if data_schema is None:
            data_schema = {}
        s_3 = replace_data_constraints(s_2, data_schema)
        return s_3

    def freeze_trainable(self) -> "TrainableIndividualOp":
        return self._configure().freeze_trainable()

    def free_hyperparams(self):
        hyperparam_schema = self.hyperparam_schema()
        if (
            "allOf" in hyperparam_schema
            and "relevantToOptimizer" in hyperparam_schema["allOf"][0]
        ):
            to_bind = hyperparam_schema["allOf"][0]["relevantToOptimizer"]
        else:
            to_bind = []
        bound = self.frozen_hyperparams()
        if bound is None:
            return set(to_bind)
        else:
            return set(to_bind) - set(bound)

    def is_frozen_trainable(self) -> bool:
        free = self.free_hyperparams()
        return len(free) == 0

    def customize_schema(
        self,
        schemas: Optional[Schema] = None,
        relevantToOptimizer: Optional[List[str]] = None,
        constraint: Union[
            Schema, JSON_TYPE, List[Union[Schema, JSON_TYPE]], None
        ] = None,
        tags: Optional[Dict] = None,
        forwards: Union[bool, List[str], None] = None,
        set_as_available: bool = False,
        **kwargs: Union[Schema, JSON_TYPE, None],
    ) -> "PlannedIndividualOp":
        return customize_schema(
            self,
            schemas,
            relevantToOptimizer,
            constraint,
            tags,
            forwards,
            set_as_available,
            **kwargs,
        )


def _mutation_warning(method_name: str) -> str:
    msg = str(
        "The `{}` method is deprecated on a trainable "
        "operator, because the learned coefficients could be "
        "accidentally overwritten by retraining. Call `{}` "
        "on the trained operator returned by `fit` instead."
    )
    return msg.format(method_name, method_name)


class TrainableIndividualOp(PlannedIndividualOp, TrainableOperator):
    def __init__(
        self,
        _lale_name,
        _lale_impl,
        _lale_schemas,
        _lale_frozen_hyperparameters=None,
        **hp,
    ):
        super().__init__(
            _lale_name, _lale_impl, _lale_schemas, _lale_frozen_hyperparameters, **hp
        )

    def set_params(self, **impl_params):
        """This implements the set_params, as per the scikit-learn convention,
        extended as documented in the module docstring"""
        return self._with_params(True, **impl_params)

    def _with_op_params(
        self, try_mutate, **impl_params: Dict[str, Any]
    ) -> "TrainableIndividualOp":
        if not try_mutate:
            return super()._with_op_params(try_mutate, **impl_params)
        hps = self.hyperparams_all()
        if hps is not None:
            hyperparams = {**hps, **impl_params}
        else:
            hyperparams = impl_params
        frozen = self.frozen_hyperparams()
        self._hyperparams = hyperparams
        if frozen:
            frozen.extend((k for k in impl_params if k not in frozen))
        else:
            self._frozen_hyperparams = list(impl_params.keys())

        if self._is_instantiated():
            # if we already have an instance impl, we need to update it
            impl = self._impl
            if hasattr(impl, "set_params"):
                new_impl = impl.set_params(**hyperparams)
                self._impl = new_impl
                self._impl_class_ = new_impl.__class__
            elif hasattr(impl, "_wrapped_model") and hasattr(
                impl._wrapped_model, "set_params"
            ):
                impl._wrapped_model.set_params(**hyperparams)
            else:
                hyper_d = {**self.get_defaults(), **hyperparams}
                self._impl = self._impl_class()(**hyper_d)

        return self

    def _clone_impl(self):
        impl_instance = self._impl_instance()
        if hasattr(impl_instance, "get_params"):
            result = sklearn.base.clone(impl_instance)
        else:
            try:
                result = copy.deepcopy(impl_instance)
            except Exception:
                impl_class = self._impl_class()
                params_all = self._get_params_all()
                result = impl_class(**params_all)
        return result

    def _trained_hyperparams(self, trained_impl) -> Optional[Dict[str, Any]]:
        hp = self.hyperparams()
        if not hp:
            return None
        # TODO: may also want to do this for other higher-order operators
        if self.class_name() != _LALE_SKL_PIPELINE:
            return hp
        names_list = [name for name, op in hp["steps"]]
        steps_list = trained_impl._pipeline.steps_list()
        trained_steps = list(zip(names_list, steps_list))
        result = {**hp, "steps": trained_steps}
        return result

    def _validate_hyperparam_data_constraints(self, X: Any, y: Any = None):
        from lale.settings import disable_hyperparams_schema_validation

        if disable_hyperparams_schema_validation:
            return
        hp_schema = self.hyperparam_schema()
        if not hasattr(self, "__has_data_constraints"):
            has_dc = has_data_constraints(hp_schema)
            self.__has_data_constraints = has_dc
        if self.__has_data_constraints:
            hp_explicit = self.hyperparams()
            hp_all = self._get_params_all()
            data_schema = fold_schema(X, y)
            hp_schema_2 = replace_data_constraints(hp_schema, data_schema)
            self._validate_hyperparams(
                hp_explicit, hp_all, hp_schema_2, self.impl_class
            )

    def fit(self, X: Any, y: Any = None, **fit_params) -> "TrainedIndividualOp":
        # logger.info("%s enter fit %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "fit")
        y = self._validate_input_schema("y", y, "fit")
        self._validate_hyperparam_data_constraints(X, y)
        filtered_fit_params = _fixup_hyperparams_dict(fit_params)

        if isinstance(self, TrainedIndividualOp):
            trainable_impl = self._impl_instance()
        else:
            trainable_impl = self._clone_impl()

        if filtered_fit_params is None:
            trained_impl = trainable_impl.fit(X, y)
        else:
            trained_impl = trainable_impl.fit(X, y, **filtered_fit_params)
        # if the trainable fit method returns None, assume that
        # the trainableshould be used as the trained impl as well
        if trained_impl is None:
            trained_impl = trainable_impl
        hps = self._trained_hyperparams(trained_impl)
        frozen: Optional[List[str]] = list(hps.keys()) if hps is not None else None
        if hps is None:
            hps = {}
        result = TrainedIndividualOp(
            self.name(),
            trained_impl,
            self._schemas,
            _lale_trained=True,
            _lale_frozen_hyperparameters=frozen,
            **hps,
        )
        if not isinstance(self, TrainedIndividualOp):
            self._trained = result
        # logger.info("%s exit  fit %s", time.asctime(), self.name())
        return result

    def partial_fit(self, X: Any, y: Any = None, **fit_params) -> "TrainedIndividualOp":
        if not self.has_method("partial_fit"):
            raise AttributeError(f"{self.name()} has no partial_fit implemented.")
        X = self._validate_input_schema("X", X, "partial_fit")
        y = self._validate_input_schema("y", y, "partial_fit")
        self._validate_hyperparam_data_constraints(X, y)
        filtered_fit_params = _fixup_hyperparams_dict(fit_params)

        # if the operator is trainable but has been trained before, use the _trained to
        # call partial fit, and update ._trained
        if hasattr(self, "_trained"):
            self._trained = self._trained.partial_fit(X, y, **fit_params)
            return self._trained
        else:
            trainable_impl = self._clone_impl()

        if filtered_fit_params is None:
            trained_impl = trainable_impl.partial_fit(X, y)
        else:
            trained_impl = trainable_impl.partial_fit(X, y, **filtered_fit_params)
        if trained_impl is None:
            trained_impl = trainable_impl
        hps = self.hyperparams_all()
        if hps is None:
            hps = {}
        result = TrainedIndividualOp(
            self.name(),
            trained_impl,
            self._schemas,
            _lale_trained=True,
            _lale_frozen_hyperparameters=self.frozen_hyperparams(),
            **hps,
        )
        if not isinstance(self, TrainedIndividualOp):
            self._trained = result
        return result

    def freeze_trained(self) -> "TrainedIndividualOp":
        """
        .. deprecated:: 0.0.0
           The `freeze_trained` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `freeze_trained`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("freeze_trained"), DeprecationWarning)
        try:
            return self._trained.freeze_trained()
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `freeze_trained`.") from exc

    def __repr__(self):
        name = self.name()
        hps = self.reduced_hyperparams()
        hyp_string: str
        if hps is None:
            hyp_string = ""
        else:
            hyp_string = lale.pretty_print.hyperparams_to_string(hps)
        return name + "(" + hyp_string + ")"

    @available_if(_impl_has("get_pipeline"))
    def get_pipeline(
        self, pipeline_name: Optional[str] = None, astype: astype_type = "lale"
    ) -> Optional[TrainableOperator]:
        """
        .. deprecated:: 0.0.0
           The `get_pipeline` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `get_pipeline`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("get_pipeline"), DeprecationWarning)
        try:
            return self._trained.get_pipeline(pipeline_name, astype)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `get_pipeline`.") from exc

    @available_if(_impl_has("summary"))
    def summary(self) -> pd.DataFrame:
        """
        .. deprecated:: 0.0.0
           The `summary` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `summary`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("summary"), DeprecationWarning)
        try:
            return self._trained.summary()
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `summary`.") from exc

    @available_if(_impl_has("transform"))
    def transform(self, X: Any, y: Any = None) -> Any:
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("transform"), DeprecationWarning)
        try:
            return self._trained.transform(X, y)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `transform`.") from exc

    @available_if(_impl_has("predict"))
    def predict(self, X=None, **predict_params) -> Any:
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict"), DeprecationWarning)
        try:
            return self._trained.predict(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict`.") from exc

    @available_if(_impl_has("predict_proba"))
    def predict_proba(self, X=None):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict_proba"), DeprecationWarning)
        try:
            return self._trained.predict_proba(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict_proba`.") from exc

    @available_if(_impl_has("decision_function"))
    def decision_function(self, X=None):
        """
        .. deprecated:: 0.0.0
           The `decision_function` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `decision_function`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("decision_function"), DeprecationWarning)
        try:
            return self._trained.decision_function(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `decision_function`.") from exc

    @available_if(_impl_has("score"))
    def score(self, X, y, **score_params) -> Any:
        """
        .. deprecated:: 0.0.0
           The `score` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `score`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("score"), DeprecationWarning)
        try:
            if score_params is None:
                return self._trained.score(X, y)
            else:
                return self._trained.score(X, y, **score_params)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `score`.") from exc

    @available_if(_impl_has("score_samples"))
    def score_samples(self, X=None):
        """
        .. deprecated:: 0.0.0
           The `score_samples` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `score_samples`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("score_samples"), DeprecationWarning)
        try:
            return self._trained.score_samples(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `score_samples`.") from exc

    @available_if(_impl_has("predict_log_proba"))
    def predict_log_proba(self, X=None):
        """
        .. deprecated:: 0.0.0
           The `predict_log_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_log_proba`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict_log_proba"), DeprecationWarning)
        try:
            return self._trained.predict_log_proba(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict_log_proba`.") from exc

    def free_hyperparams(self) -> Set[str]:
        hyperparam_schema = self.hyperparam_schema()
        to_bind: List[str]
        if (
            "allOf" in hyperparam_schema
            and "relevantToOptimizer" in hyperparam_schema["allOf"][0]
        ):
            to_bind = hyperparam_schema["allOf"][0]["relevantToOptimizer"]
        else:
            to_bind = []
        bound = self.frozen_hyperparams()
        if bound is None:
            return set(to_bind)
        else:
            return set(to_bind) - set(bound)

    def _freeze_trainable_bindings(self) -> Dict[str, Any]:
        old_bindings = self.hyperparams_all()
        if old_bindings is None:
            old_bindings = {}
        free = self.free_hyperparams()
        defaults: Mapping[str, Any] = self.get_defaults()
        new_bindings: Dict[str, Any] = {name: defaults[name] for name in free}
        bindings: Dict[str, Any] = {**old_bindings, **new_bindings}
        return bindings

    def freeze_trainable(self) -> "TrainableIndividualOp":
        bindings = self._freeze_trainable_bindings()
        result = self._configure(**bindings)
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        return result

    def transform_schema(self, s_X: JSON_TYPE):
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return {}
        if self.has_method("transform_schema"):
            try:
                return self._impl_instance().transform_schema(s_X)
            except BaseException as exc:
                raise ValueError(
                    f"unexpected error in {self.name()}.transform_schema({lale.pretty_print.to_string(s_X)}"
                ) from exc
        else:
            return super().transform_schema(s_X)

    def input_schema_fit(self) -> JSON_TYPE:
        if self.has_method("input_schema_fit"):
            return self._impl_instance().input_schema_fit()
        else:
            return super().input_schema_fit()

    def customize_schema(
        self,
        schemas: Optional[Schema] = None,
        relevantToOptimizer: Optional[List[str]] = None,
        constraint: Union[
            Schema, JSON_TYPE, List[Union[Schema, JSON_TYPE]], None
        ] = None,
        tags: Optional[Dict] = None,
        forwards: Union[bool, List[str], None] = None,
        set_as_available: bool = False,
        **kwargs: Union[Schema, JSON_TYPE, None],
    ) -> "TrainableIndividualOp":
        return customize_schema(
            self,
            schemas,
            relevantToOptimizer,
            constraint,
            tags,
            forwards,
            set_as_available,
            **kwargs,
        )

    def convert_to_trained(self) -> "TrainedIndividualOp":
        trained_op = TrainedIndividualOp(
            _lale_name=self._name,
            _lale_impl=self.impl,
            _lale_schemas=self._schemas,
            _lale_frozen_hyperparameters=self.frozen_hyperparams(),
            _lale_trained=True,
        )
        if hasattr(self, "_frozen_trained"):
            trained_op._frozen_trained = self._frozen_trained
        if hasattr(self, "_hyperparams"):
            trained_op._hyperparams = self._hyperparams
        return trained_op


class TrainedIndividualOp(TrainableIndividualOp, TrainedOperator):
    _frozen_trained: bool

    def __new__(cls, *args, _lale_trained=False, _lale_impl=None, **kwargs):
        if (
            "_lale_name" not in kwargs
            or _lale_trained
            or (_lale_impl is not None and not hasattr(_lale_impl, "fit"))
        ):
            obj = super().__new__(TrainedIndividualOp)  # type: ignore
            return obj
        else:
            # unless _lale_trained=True, we actually want to return a Trainable
            obj = super().__new__(TrainableIndividualOp)  # type: ignore
            # apparently python does not call __init__ if the type returned is not the
            # expected type
            obj.__init__(*args, **kwargs)
            return obj

    def __init__(
        self,
        _lale_name,
        _lale_impl,
        _lale_schemas,
        _lale_frozen_hyperparameters=None,
        _lale_trained=False,
        **hp,
    ):
        super().__init__(
            _lale_name, _lale_impl, _lale_schemas, _lale_frozen_hyperparameters, **hp
        )
        self._frozen_trained = not self.has_method("fit")

    def __call__(self, *args, **kwargs) -> "TrainedIndividualOp":
        filtered_kwargs_params = _fixup_hyperparams_dict(kwargs)

        trainable = self._configure(*args, **filtered_kwargs_params)
        hps = trainable.hyperparams_all()
        if hps is None:
            hps = {}
        instance = TrainedIndividualOp(
            trainable._name,
            trainable._impl,
            trainable._schemas,
            _lale_trained=True,
            _lale_frozen_hyperparameters=trainable.frozen_hyperparams(),
            **hps,
        )
        return instance

    def fit(self, X: Any, y: Any = None, **fit_params) -> "TrainedIndividualOp":
        if self.has_method("fit") and not self.is_frozen_trained():
            filtered_fit_params = _fixup_hyperparams_dict(fit_params)
            try:
                return super().fit(X, y, **filtered_fit_params)
            except AttributeError:
                return self  # for Project with static columns after clone()
        else:
            return self

    @available_if(_impl_has("transform"))
    def transform(self, X: Any, y: Any = None) -> Any:
        """Transform the data.

        Parameters
        ----------
        X :
            Features; see input_transform schema of the operator.

        y: None

        Returns
        -------
        result :
            Transformed features; see output_transform schema of the operator.
        """
        # logger.info("%s enter transform %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "transform")
        if "y" in [
            required_property.lower()
            for required_property in self.input_schema_transform().get("required", [])
        ]:
            y = self._validate_input_schema("y", y, "transform")
            raw_result = self._impl_instance().transform(X, y)
        else:
            raw_result = self._impl_instance().transform(X)
        result = self._validate_output_schema(raw_result, "transform")
        # logger.info("%s exit  transform %s", time.asctime(), self.name())
        return result

    @available_if(_impl_has("transform_X_y"))
    def transform_X_y(self, X: Any, y: Any) -> Any:
        """Transform the data and target.

        Parameters
        ----------
        X :
            Features; see input_transform schema of the operator.

        y :
            target; see input_transform schema of the operator.

        Returns
        -------
        result :
            Transformed features and target; see output_transform schema of the operator.
        """
        X = self._validate_input_schema("X", X, "transform_X_y")
        y = self._validate_input_schema("y", y, "transform_X_y")
        output_X, output_y = self._impl_instance().transform_X_y(X, y)
        output_X, output_y = self._validate_output_schema(
            (output_X, output_y), "transform_X_y"
        )
        return output_X, output_y

    def _predict(self, X, **predict_params):
        X = self._validate_input_schema("X", X, "predict")

        raw_result = self._impl_instance().predict(X, **predict_params)
        result = self._validate_output_schema(raw_result, "predict")
        return result

    @available_if(_impl_has("predict"))
    def predict(self, X: Any = None, **predict_params) -> Any:
        """Make predictions.

        Parameters
        ----------
        X :
            Features; see input_predict schema of the operator.
        predict_params:
            Additional parameters that should be passed to the predict method

        Returns
        -------
        result :
            Predictions; see output_predict schema of the operator.
        """
        # logger.info("%s enter predict %s", time.asctime(), self.name())
        result = self._predict(X, **predict_params)
        # logger.info("%s exit  predict %s", time.asctime(), self.name())
        if isinstance(result, NDArrayWithSchema):
            return strip_schema(result)  # otherwise scorers return zero-dim array
        return result

    @available_if(_impl_has("predict_proba"))
    def predict_proba(self, X: Any = None):
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        # logger.info("%s enter predict_proba %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "predict_proba")
        raw_result = self._impl_instance().predict_proba(X)
        result = self._validate_output_schema(raw_result, "predict_proba")
        # logger.info("%s exit  predict_proba %s", time.asctime(), self.name())
        return result

    @available_if(_impl_has("decision_function"))
    def decision_function(self, X: Any = None):
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        # logger.info("%s enter decision_function %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "decision_function")
        raw_result = self._impl_instance().decision_function(X)
        result = self._validate_output_schema(raw_result, "decision_function")
        # logger.info("%s exit  decision_function %s", time.asctime(), self.name())
        return result

    @available_if(_impl_has("score"))
    def score(self, X: Any, y: Any, **score_params) -> Any:
        """Performance evaluation with a default metric.

        Parameters
        ----------
        X :
            Features.
        y:
            Ground truth labels.
        score_params:
            Any additional parameters expected by the score function of
            the underlying operator.
        Returns
        -------
        score :
            performance metric value
        """
        # Use the input schema of predict as in most cases it applies to score as well.
        X = self._validate_input_schema("X", X, "predict")
        if score_params is None:
            result = self._impl_instance().score(X, y)
        else:
            result = self._impl_instance().score(X, y, **score_params)
        # We skip output validation for score for now
        return result

    @available_if(_impl_has("score_samples"))
    def score_samples(self, X: Any = None):
        """Scores for each sample in X. The type of scores depends on the operator.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            scores per sample.
        """
        X = self._validate_input_schema("X", X, "score_samples")
        raw_result = self._impl_instance().score_samples(X)
        result = self._validate_output_schema(raw_result, "score_samples")
        return result

    @available_if(_impl_has("predict_log_proba"))
    def predict_log_proba(self, X: Any = None):
        """Predicted class log-probabilities for X.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            Class log probabilities.
        """
        X = self._validate_input_schema("X", X, "predict_log_proba")
        raw_result = self._impl_instance().predict_log_proba(X)
        result = self._validate_output_schema(raw_result, "predict_log_proba")
        return result

    def freeze_trainable(self) -> "TrainedIndividualOp":
        result = copy.deepcopy(self)
        new_bindings = self._freeze_trainable_bindings()
        result._hyperparams = new_bindings
        result._frozen_hyperparams = list(new_bindings)
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        assert isinstance(result, TrainedIndividualOp)
        return result

    def is_frozen_trained(self) -> bool:
        return self._frozen_trained

    def freeze_trained(self) -> "TrainedIndividualOp":
        if self.is_frozen_trained():
            return self
        result = copy.deepcopy(self)
        result._frozen_trained = True
        assert result.is_frozen_trained()
        return result

    @overload
    def get_pipeline(
        self, pipeline_name: None = None, astype: astype_type = "lale"
    ) -> Optional[TrainedOperator]: ...

    @overload
    def get_pipeline(  # pylint:disable=signature-differs
        self, pipeline_name: str, astype: astype_type = "lale"
    ) -> Optional[TrainableOperator]: ...

    @available_if(_impl_has("get_pipeline"))
    def get_pipeline(self, pipeline_name=None, astype: astype_type = "lale"):
        result = self._impl_instance().get_pipeline(pipeline_name, astype)
        return result

    @available_if(_impl_has("summary"))
    def summary(self) -> pd.DataFrame:
        return self._impl_instance().summary()

    def customize_schema(
        self,
        schemas: Optional[Schema] = None,
        relevantToOptimizer: Optional[List[str]] = None,
        constraint: Union[
            Schema, JSON_TYPE, List[Union[Schema, JSON_TYPE]], None
        ] = None,
        tags: Optional[Dict] = None,
        forwards: Union[bool, List[str], None] = None,
        set_as_available: bool = False,
        **kwargs: Union[Schema, JSON_TYPE, None],
    ) -> "TrainedIndividualOp":
        return customize_schema(
            self,
            schemas,
            relevantToOptimizer,
            constraint,
            tags,
            forwards,
            set_as_available,
            **kwargs,
        )

    def partial_fit(self, X: Any, y: Any = None, **fit_params) -> "TrainedIndividualOp":
        if not self.has_method("partial_fit"):
            raise AttributeError(f"{self.name()} has no partial_fit implemented.")
        X = self._validate_input_schema("X", X, "partial_fit")
        y = self._validate_input_schema("y", y, "partial_fit")
        self._validate_hyperparam_data_constraints(X, y)
        filtered_fit_params = _fixup_hyperparams_dict(fit_params)

        # Since this is a trained operator and we are calling partial_fit,
        # we allow the trained op to be mutated by using the same impl to
        # call partial_fit
        trainable_impl = self.shallow_impl
        if filtered_fit_params is None:
            trained_impl = trainable_impl.partial_fit(X, y)
        else:
            trained_impl = trainable_impl.partial_fit(X, y, **filtered_fit_params)
        if trained_impl is None:
            trained_impl = trainable_impl
        self._impl = trained_impl
        return self


_all_available_operators: List[PlannedOperator] = []


def wrap_operator(impl) -> Operator:
    if isinstance(impl, Operator):
        return impl
    else:
        return make_operator(impl)


# variant of make_operator for impls that are already trained (don't have a fit method)
def make_pretrained_operator(
    impl, schemas=None, name: Optional[str] = None
) -> TrainedIndividualOp:
    x = make_operator(impl, schemas, name)
    assert isinstance(x, TrainedIndividualOp)
    return x


def get_op_from_lale_lib(impl_class, wrapper_modules=None) -> Optional[IndividualOp]:
    assert inspect.isclass(impl_class)
    assert not issubclass(impl_class, Operator)
    assert hasattr(impl_class, "predict") or hasattr(impl_class, "transform")
    result = None
    if impl_class.__module__.startswith("lale.lib"):
        assert impl_class.__name__.endswith("Impl"), impl_class.__name__
        assert impl_class.__name__.startswith("_"), impl_class.__name__
        module = importlib.import_module(impl_class.__module__)
        class_name = impl_class.__name__[1 : -len("Impl")]
        result = getattr(module, class_name)
    else:
        try:
            module_name = impl_class.__module__.split(".")[0]
            module = importlib.import_module("lale.lib." + module_name)
            result = getattr(module, impl_class.__name__)
        except (ModuleNotFoundError, AttributeError):
            try:
                module = importlib.import_module("lale.lib.autogen")
                result = getattr(module, impl_class.__name__)
            except (ModuleNotFoundError, AttributeError):
                if wrapper_modules is not None:
                    for wrapper_module in wrapper_modules:
                        try:
                            module = importlib.import_module(wrapper_module)
                            result = getattr(module, impl_class.__name__)
                            if result is not None:
                                break
                        except (ModuleNotFoundError, AttributeError):
                            pass
                    else:
                        result = None
    if result is not None:
        result._check_schemas()
    return result


def get_lib_schemas(impl_class) -> Optional[JSON_TYPE]:
    operator = get_op_from_lale_lib(impl_class)
    return None if operator is None else operator._schemas


def make_operator(
    impl, schemas=None, name: Optional[str] = None, set_as_available: bool = True
) -> PlannedIndividualOp:
    if name is None:
        name = assignee_name(level=2)
        if name is None:
            if inspect.isclass(impl):
                n: str = impl.__name__
                if n.startswith("_"):
                    n = n[1:]
                if n.endswith("Impl"):
                    n = n[: -len("Impl")]
                name = n
            else:
                name = "Unknown"
    if schemas is None:
        if isinstance(impl, IndividualOp):
            schemas = impl._schemas
        elif inspect.isclass(impl):
            schemas = get_lib_schemas(impl)
        else:
            schemas = get_lib_schemas(impl.__class__)
    if inspect.isclass(impl):
        if hasattr(impl, "fit"):
            operatorObj = PlannedIndividualOp(
                name, impl, schemas, _lale_frozen_hyperparameters=None
            )
        else:
            operatorObj = TrainedIndividualOp(
                name,
                impl,
                schemas,
                _lale_trained=True,
                _lale_frozen_hyperparameters=None,
            )
    else:
        hps: Dict[str, Any] = {}
        frozen: Optional[List[str]] = None
        impl_get_params = getattr(impl, "get_params", None)
        if impl_get_params is not None:
            hps = impl_get_params(deep=False)
            frozen = list(hps.keys())

        if hasattr(impl, "fit"):
            operatorObj = TrainableIndividualOp(
                name, impl, schemas, _lale_frozen_hyperparameters=frozen, **hps
            )
        else:
            operatorObj = TrainedIndividualOp(
                name,
                impl,
                schemas,
                _lale_trained=True,
                _lale_frozen_hyperparameters=frozen,
                **hps,
            )

    operatorObj._check_schemas()

    if set_as_available:
        _all_available_operators.append(operatorObj)
    return operatorObj


def get_available_operators(
    tag: str, more_tags: Optional[AbstractSet[str]] = None
) -> List[PlannedOperator]:
    singleton = set([tag])
    tags = singleton if (more_tags is None) else singleton.union(more_tags)

    def filter_by_tags(op):
        tags_dict = op.get_tags()
        if tags_dict is None:
            return False
        tags_set = {tag for prefix in tags_dict for tag in tags_dict[prefix]}
        return tags.issubset(tags_set)

    return [op for op in _all_available_operators if filter_by_tags(op)]


def get_available_estimators(
    tags: Optional[AbstractSet[str]] = None,
) -> List[PlannedOperator]:
    return get_available_operators("estimator", tags)


def get_available_transformers(
    tags: Optional[AbstractSet[str]] = None,
) -> List[PlannedOperator]:
    return get_available_operators("transformer", tags)


OpType_co = TypeVar("OpType_co", bound=Operator, covariant=True)


class BasePipeline(Operator, Generic[OpType_co]):
    """
    This is a concrete class that can instantiate a new pipeline operator and provide access to its meta data.
    """

    _steps: List[OpType_co]
    _preds: Dict[OpType_co, List[OpType_co]]
    _cached_preds: Optional[Dict[int, List[int]]]
    _name: str

    def _steps_to_indices(self) -> Dict[OpType_co, int]:
        return {op: i for i, op in enumerate(self._steps)}

    def _preds_to_indices(self) -> Dict[int, List[int]]:
        step_map = self._steps_to_indices()
        return {
            step_map[k]: ([step_map[v] for v in vs]) for (k, vs) in self._preds.items()
        }

    def _get_preds_indices(self) -> Dict[int, List[int]]:
        p: Dict[int, List[int]]
        if self._cached_preds is None:
            p = self._preds_to_indices()
            self._cached_preds = p
        else:
            p = self._cached_preds
        return p

    @property
    def _estimator_type(self):
        estimator = self._final_individual_op
        if estimator is not None:
            return estimator._estimator_type
        else:
            raise ValueError(
                "Cannot determine the _estimator_type, since this pipeline does not have a unique final operator"
            )

    @classmethod
    def _indices_to_preds(
        cls, _steps: List[OpType_co], _pred_indices: Dict[int, List[int]]
    ) -> Dict[OpType_co, List[OpType_co]]:
        return {
            _steps[k]: ([_steps[v] for v in vs]) for (k, vs) in _pred_indices.items()
        }

    def get_params(self, deep: Union[bool, Literal[0]] = True) -> Dict[str, Any]:
        """
        If deep is False, additional '_lale_XXX' fields are added to support
        cloning.  If these are not desires, deep=0 can be used to disable this
        """
        out: Dict[str, Any] = {}
        out["steps"] = self._steps
        if deep is False:
            out["_lale_preds"] = self._get_preds_indices()

        indices: Dict[str, int] = {}

        def make_indexed(name: str) -> str:
            idx = 0
            if name in indices:
                idx = indices[name] + 1
                indices[name] = idx
            else:
                indices[name] = 0
            return make_indexed_name(name, idx)

        if deep:
            for op in self._steps:
                name = make_indexed(op.name())
                nested_params = op.get_params(deep=deep)
                if nested_params:
                    out.update(nest_HPparams(name, nested_params))
        return out

    def set_params(self, **impl_params):
        """This implements the set_params, as per the scikit-learn convention,
        extended as documented in the module docstring"""
        return self._with_params(True, **impl_params)

    def _with_params(
        self, try_mutate: bool, **impl_params
    ) -> "BasePipeline[OpType_co]":
        steps = self.steps_list()
        main_params, partitioned_sub_params = partition_sklearn_params(impl_params)
        assert not main_params, f"Unexpected non-nested arguments {main_params}"
        found_names: Dict[str, int] = {}
        step_map: Dict[OpType_co, OpType_co] = {}
        for s in steps:
            name = s.name()
            name_index = 0
            params: Dict[str, Any] = {}
            if name in found_names:
                name_index = found_names[name] + 1
                found_names[name] = name_index
                uname = make_indexed_name(name, name_index)
                params = partitioned_sub_params.get(uname, params)
            else:
                found_names[name] = 0
                uname = make_degen_indexed_name(name, 0)
                if uname in partitioned_sub_params:
                    params = partitioned_sub_params[uname]
                    assert name not in partitioned_sub_params
                else:
                    params = partitioned_sub_params.get(name, params)
            new_s = s._with_params(try_mutate, **params)
            if s != new_s:
                # getting this to statically type check would be very complicated
                # if even possible
                step_map[s] = new_s  # type: ignore
        # make sure that no parameters were passed in for operations
        # that are not actually part of this pipeline
        for k in partitioned_sub_params:
            n, i = get_name_and_index(k)
            assert n in found_names and i <= found_names[n]

        if try_mutate:
            if step_map:
                self._subst_steps(step_map)

            pipeline_graph_class = _pipeline_graph_class(self.steps_list())
            self.__class__ = pipeline_graph_class  # type: ignore
            return self
        else:
            needs_copy = False
            if step_map:
                needs_copy = True
            else:
                pipeline_graph_class = _pipeline_graph_class(self.steps_list())
                if pipeline_graph_class != self.__class__:  # type: ignore
                    needs_copy = True
            if needs_copy:
                # it may be better practice to change the steps/edges ahead of time
                # and then create the correct class
                op_copy = make_pipeline_graph(self.steps_list(), self.edges(), ordered=True)  # type: ignore
                op_copy._subst_steps(step_map)

                pipeline_graph_class = _pipeline_graph_class(op_copy.steps_list())
                op_copy.__class__ = pipeline_graph_class  # type: ignore
                return op_copy
            else:
                return self

    def __init__(
        self,
        steps: List[OpType_co],
        edges: Optional[Iterable[Tuple[OpType_co, OpType_co]]] = None,
        _lale_preds: Optional[
            Union[Dict[int, List[int]], Dict[OpType_co, List[OpType_co]]]
        ] = None,
        ordered: bool = False,
    ) -> None:
        self._name = "pipeline_" + str(id(self))
        self._preds = {}
        for step in steps:
            assert isinstance(step, Operator)
        if _lale_preds is not None:
            # this is a special case that is meant for use with cloning
            # if preds is set, we assume that it is ordered as well
            assert edges is None
            self._steps = steps
            if _lale_preds:
                # TODO: improve typing situation
                keys: Iterable[Any] = _lale_preds.keys()
                first_key = next(iter(keys))
                if isinstance(first_key, int):
                    self._preds = self._indices_to_preds(steps, _lale_preds)  # type: ignore
                    self._cached_preds = _lale_preds  # type: ignore
                else:
                    self._preds = _lale_preds  # type: ignore
                    self._cached_preds = None  # type: ignore
            else:
                self._cached_preds = _lale_preds  # type: ignore
            return
        self._cached_preds = None
        if edges is None:
            # Which means there is a linear pipeline #TODO:Test extensively with clone and get_params
            # This constructor is mostly called due to cloning. Make sure the objects are kept the same.
            self.__constructor_for_cloning(steps)
        else:
            self._steps = []

            for step in steps:
                if step in self._steps:
                    raise ValueError(
                        f"Same instance of {step.name()} already exists in the pipeline. "
                        f"This is not allowed."
                    )
                if isinstance(step, BasePipeline):
                    # PIPELINE_TYPE_INVARIANT_NOTE
                    # we use tstep (typed step) here to help pyright
                    # with some added information we have:
                    # Since the step is an OpType, if it is a pipeline,
                    # then its steps must all be at least OpType as well
                    # this invariant is not expressible in the type system due to
                    # the open world assumption, but is intended to hold
                    tstep: BasePipeline[OpType_co] = step

                    # Flatten out the steps and edges
                    self._steps.extend(tstep.steps_list())
                    # from step's edges, find out all the source and sink nodes
                    source_nodes = [
                        dst
                        for dst in tstep.steps_list()
                        if (step._preds[dst] is None or step._preds[dst] == [])
                    ]
                    sink_nodes = tstep._find_sink_nodes()
                    # Now replace the edges to and from the inner pipeline to to and from source and sink nodes respectively
                    new_edges: List[Tuple[OpType_co, OpType_co]] = tstep.edges()
                    # list comprehension at the cost of iterating edges thrice
                    new_edges.extend(
                        [
                            (node, edge[1])
                            for edge in edges
                            if edge[0] == tstep
                            for node in sink_nodes
                        ]
                    )
                    new_edges.extend(
                        [
                            (edge[0], node)
                            for edge in edges
                            if edge[1] == tstep
                            for node in source_nodes
                        ]
                    )
                    new_edges.extend(
                        edge for edge in edges if tstep not in (edge[0], edge[1])
                    )
                    edges = new_edges
                else:
                    self._steps.append(step)
            self._preds = {step: [] for step in self._steps}
            for src, dst in edges:
                self._preds[dst].append(src)  # type: ignore
            if not ordered:
                self.__sort_topologically()
            assert self.__is_in_topological_order()

    def __constructor_for_cloning(self, steps: List[OpType_co]):
        edges: List[Tuple[OpType_co, OpType_co]] = []
        prev_op: Optional[OpType_co] = None
        # This is due to scikit base's clone method that needs the same list object
        self._steps = steps
        prev_leaves: List[OpType_co]
        curr_roots: List[OpType_co]

        for curr_op in self._steps:
            if isinstance(prev_op, BasePipeline):
                # using tprev_op as per PIPELINE_TYPE_INVARIANT_NOTE above
                tprev_op: BasePipeline[OpType_co] = prev_op
                prev_leaves = tprev_op._find_sink_nodes()
            else:
                prev_leaves = [] if prev_op is None else [prev_op]
            prev_op = curr_op

            if isinstance(curr_op, BasePipeline):
                # using tcurr_op as per PIPELINE_TYPE_INVARIANT_NOTE above
                tcurr_op: BasePipeline[OpType_co] = curr_op
                curr_roots = tcurr_op._find_source_nodes()
                self._steps.extend(tcurr_op.steps_list())
                edges.extend(tcurr_op.edges())
            else:
                curr_roots = [curr_op]
            edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])

        seen_steps: List[OpType_co] = []
        for step in self._steps:
            if step in seen_steps:
                raise ValueError(
                    f"Same instance of {step.name()} already exists in the pipeline. "
                    f"This is not allowed."
                )
            seen_steps.append(step)
        self._preds = {step: [] for step in self._steps}
        for src, dst in edges:
            self._preds[dst].append(src)
        # Since this case is only allowed for linear pipelines, it is always
        # expected to be in topological order
        assert self.__is_in_topological_order()

    def edges(self) -> List[Tuple[OpType_co, OpType_co]]:
        return [(src, dst) for dst in self._steps for src in self._preds[dst]]

    def __is_in_topological_order(self) -> bool:
        seen: Dict[OpType_co, bool] = {}
        for operator in self._steps:
            for pred in self._preds[operator]:
                if pred not in seen:
                    return False
            seen[operator] = True
        return True

    def steps_list(self) -> List[OpType_co]:
        return self._steps

    @property
    def steps(self) -> List[Tuple[str, OpType_co]]:
        """This is meant to function similarly to the scikit-learn steps property
        and for linear pipelines, should behave the same
        """
        return [(s.name(), s) for s in self._steps]

    def _subst_steps(self, m: Dict[OpType_co, OpType_co]) -> None:
        if m:
            # for i, s in enumerate(self._steps):
            #     self._steps[i] = m.get(s,s)
            self._steps = [m.get(s, s) for s in self._steps]
            self._preds = {
                m.get(k, k): [m.get(s, s) for s in v] for k, v in self._preds.items()
            }

    def __sort_topologically(self) -> None:
        class state(enumeration.Enum):
            TODO = (enumeration.auto(),)
            DOING = (enumeration.auto(),)
            DONE = enumeration.auto()

        states: Dict[OpType_co, state] = {op: state.TODO for op in self._steps}
        result: List[OpType_co] = []

        # Since OpType is covariant, this is disallowed by mypy for safety
        # in this case it is safe, since while the value of result will be written
        # into _steps, all the values in result came from _steps originally
        def dfs(operator: OpType_co) -> None:  # type: ignore
            if states[operator] is state.DONE:
                return
            if states[operator] is state.DOING:
                raise ValueError("Cycle detected.")
            states[operator] = state.DOING
            for pred in self._preds[operator]:
                dfs(pred)
            states[operator] = state.DONE
            result.append(operator)

        for operator in self._steps:
            if states[operator] is state.TODO:
                dfs(operator)
        self._steps = result

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator imnplementations are compatible"""
        if not isinstance(other, BasePipeline):
            return False
        my_steps = self.steps_list()
        other_steps = other.steps_list()
        if len(my_steps) != len(other_steps):
            return False

        for m, o in zip(my_steps, other_steps):
            if not m._has_same_impl(o):
                return False
        return True

    def _find_sink_nodes(self) -> List[OpType_co]:
        is_sink = {s: True for s in self.steps_list()}
        for src, _ in self.edges():
            is_sink[src] = False
        result = [s for s in self.steps_list() if is_sink[s]]
        return result

    def _find_source_nodes(self) -> List[OpType_co]:
        is_source = {s: True for s in self.steps_list()}
        for _, dst in self.edges():
            is_source[dst] = False
        result = [s for s in self.steps_list() if is_source[s]]
        return result

    @overload
    def _validate_or_transform_schema(
        self, X, *, y: Optional[Any], validate: Literal[False]
    ) -> JSON_TYPE: ...

    @overload
    def _validate_or_transform_schema(
        self, X, *, y: Optional[Any] = None, validate: Literal[False]
    ) -> JSON_TYPE: ...

    @overload
    def _validate_or_transform_schema(
        self, X, y: Optional[Any] = None, validate: Literal[True] = True
    ) -> None: ...

    def _validate_or_transform_schema(
        self, X: Any, y: Any = None, validate=True
    ) -> Optional[JSON_TYPE]:
        def combine_schemas(schemas):
            n_datasets = len(schemas)
            if n_datasets == 1:
                result = schemas[0]
            else:
                result = {
                    "type": "array",
                    "minItems": n_datasets,
                    "maxItems": n_datasets,
                    "items": [_to_schema(i) for i in schemas],
                }
            return result

        outputs: Dict[OpType_co, Any] = {}
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                input_X, input_y = X, y
            else:
                input_X = combine_schemas([outputs[pred][0] for pred in preds])
                input_y = outputs[preds[0]][1]
            if validate:
                operator.validate_schema(X=input_X, y=input_y)
            if operator.has_method("transform_X_y"):
                output_Xy = operator.output_schema_transform_X_y()
                output_X, output_y = output_Xy["items"]
            else:
                output_X = operator.transform_schema(input_X)
                output_y = input_y
            outputs[operator] = output_X, output_y
        if validate:
            return None
        else:
            sinks = self._find_sink_nodes()
            pipeline_outputs = [outputs[sink][0] for sink in sinks]
            return combine_schemas(pipeline_outputs)

    def validate_schema(self, X: Any, y: Any = None):
        self._validate_or_transform_schema(X, y, validate=True)

    def transform_schema(self, s_X: JSON_TYPE) -> JSON_TYPE:
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return {}
        else:
            return self._validate_or_transform_schema(s_X, validate=False)

    def input_schema_fit(self) -> JSON_TYPE:
        sources = self._find_source_nodes()
        pipeline_inputs = [source.input_schema_fit() for source in sources]
        result = join_schemas(*pipeline_inputs)
        return result

    def is_supervised(self) -> bool:
        s = self.steps_list()
        if len(s) == 0:
            return False
        return self.steps_list()[-1].is_supervised()

    def remove_last(self, inplace: bool = False) -> "BasePipeline[OpType_co]":
        sink_nodes = self._find_sink_nodes()
        if len(sink_nodes) > 1:
            raise ValueError(
                "This pipeline has more than 1 sink nodes, can not remove last step meaningfully."
            )
        if not inplace:
            modified_pipeline = copy.deepcopy(self)
            old_clf = modified_pipeline._steps[-1]
            modified_pipeline._steps.remove(old_clf)
            del modified_pipeline._preds[old_clf]
            return modified_pipeline
        else:
            old_clf = self._steps[-1]
            self._steps.remove(old_clf)
            del self._preds[old_clf]
            return self

    def get_last(self) -> Optional[OpType_co]:
        sink_nodes = self._find_sink_nodes()
        if len(sink_nodes) > 1:
            return None
        else:
            old_clf = self._steps[-1]
            return old_clf

    def export_to_sklearn_pipeline(self):
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import make_pipeline as sklearn_make_pipeline

        from lale.lib.lale.no_op import NoOp
        from lale.lib.rasl.concat_features import ConcatFeatures
        from lale.lib.rasl.relational import Relational

        def convert_nested_objects(node):
            for element in dir(node):  # Looking at only 1 level for now.
                try:
                    value = getattr(node, element)
                    if isinstance(value, IndividualOp):
                        if isinstance(value.shallow_impl, sklearn.base.BaseEstimator):
                            setattr(node, element, value.shallow_impl)
                        if hasattr(value.shallow_impl, "_wrapped_model"):
                            # node is a higher order operator
                            setattr(node, element, value.shallow_impl._wrapped_model)

                    stripped = strip_schema(value)
                    if value is stripped:
                        continue
                    setattr(node, element, stripped)
                except BaseException:
                    # This is an optional processing, so if there is any exception, continue.
                    # For example, some scikit-learn classes will fail at getattr because they have
                    # that property defined conditionally.
                    pass

        def create_pipeline_from_sink_node(sink_node):
            # Ensure that the pipeline is either linear or has a "union followed by concat" construct
            # Translate the "union followed by concat" constructs to "featureUnion"
            # Inspect the node and convert any data with schema objects to original data types
            if isinstance(sink_node, OperatorChoice):
                raise ValueError(
                    f"A pipeline that has an OperatorChoice can not be converted to "
                    f" a scikit-learn pipeline:{self.to_json()}"
                )
            if sink_node.impl_class == Relational.impl_class:
                return None
            convert_nested_objects(sink_node._impl)
            if sink_node.impl_class == ConcatFeatures.impl_class:
                list_of_transformers = []
                for pred in self._preds[sink_node]:
                    pred_transformer = create_pipeline_from_sink_node(pred)
                    list_of_transformers.append(
                        (
                            pred.name() + "_" + str(id(pred)),
                            (
                                sklearn_make_pipeline(*pred_transformer)
                                if isinstance(pred_transformer, list)
                                else pred_transformer
                            ),
                        )
                    )
                return FeatureUnion(list_of_transformers)
            else:
                preds = self._preds[sink_node]
                if preds is not None and len(preds) > 1:
                    raise ValueError(
                        f"A pipeline graph that has operators other than ConcatFeatures with "
                        f"multiple incoming edges is not a valid scikit-learn pipeline:{self.to_json()}"
                    )

                if hasattr(sink_node.shallow_impl, "_wrapped_model"):
                    sklearn_op = sink_node.shallow_impl._wrapped_model
                    convert_nested_objects(
                        sklearn_op
                    )  # This case needs one more level of conversion
                else:
                    sklearn_op = sink_node.shallow_impl
                sklearn_op = copy.deepcopy(sklearn_op)
                if preds is None or len(preds) == 0:
                    return sklearn_op
                else:
                    output_pipeline_steps = []
                    previous_sklearn_op = create_pipeline_from_sink_node(preds[0])
                    if previous_sklearn_op is not None and not isinstance(
                        previous_sklearn_op, NoOp.impl_class
                    ):
                        if isinstance(previous_sklearn_op, list):
                            output_pipeline_steps = previous_sklearn_op
                        else:
                            output_pipeline_steps.append(previous_sklearn_op)
                    if not isinstance(
                        sklearn_op, NoOp.impl_class
                    ):  # Append the current op only if not NoOp
                        output_pipeline_steps.append(sklearn_op)
                    return output_pipeline_steps

        sklearn_steps_list = []
        # Finding the sink node so that we can do a backward traversal
        sink_nodes = self._find_sink_nodes()
        # For a trained pipeline that is scikit compatible, there should be only one sink node
        if len(sink_nodes) != 1:
            raise ValueError(
                f"A pipeline graph that ends with more than one estimator is not a"
                f" valid scikit-learn pipeline:{self.to_json()}"
            )

        sklearn_steps_list = create_pipeline_from_sink_node(sink_nodes[0])
        # not checking for isinstance(sklearn_steps_list, NoOp) here as there is no valid sklearn pipeline with just one NoOp.
        try:
            sklearn_pipeline = (
                sklearn_make_pipeline(*sklearn_steps_list)
                if isinstance(sklearn_steps_list, list)
                else sklearn_make_pipeline(sklearn_steps_list)
            )
        except TypeError as exc:
            raise TypeError(
                "Error creating a scikit-learn pipeline, most likely because the steps are not scikit compatible."
            ) from exc
        return sklearn_pipeline

    def is_classifier(self) -> bool:
        sink_nodes = self._find_sink_nodes()
        for op in sink_nodes:
            if not op.is_classifier():
                return False
        return True

    def get_defaults(self) -> Dict[str, Any]:
        defaults_list: Iterable[Dict[str, Any]] = (
            nest_HPparams(s.name(), s.get_defaults()) for s in self.steps_list()
        )

        # TODO: could this just be dict(defaults_list)
        defaults: Dict[str, Any] = {}
        for d in defaults_list:
            defaults.update(d)

        return defaults

    @property
    def _final_individual_op(self) -> Optional["IndividualOp"]:
        op = self.get_last()
        if op is None:
            return None
        else:
            return op._final_individual_op


PlannedOpType_co = TypeVar("PlannedOpType_co", bound=PlannedOperator, covariant=True)


class PlannedPipeline(BasePipeline[PlannedOpType_co], PlannedOperator):
    def __init__(
        self,
        steps: List[PlannedOpType_co],
        edges: Optional[Iterable[Tuple[PlannedOpType_co, PlannedOpType_co]]] = None,
        _lale_preds: Optional[Dict[int, List[int]]] = None,
        ordered: bool = False,
    ) -> None:
        super().__init__(steps, edges=edges, _lale_preds=_lale_preds, ordered=ordered)

    # give it a more precise type: if the input is a pipeline, the output is as well
    def auto_configure(
        self, X: Any, y: Any = None, optimizer=None, cv=None, scoring=None, **kwargs
    ) -> "TrainedPipeline":
        trained = super().auto_configure(
            X, y=y, optimizer=optimizer, cv=cv, scoring=scoring, **kwargs
        )
        assert isinstance(trained, TrainedPipeline)
        return trained

    def remove_last(self, inplace: bool = False) -> "PlannedPipeline[PlannedOpType_co]":
        pipe = super().remove_last(inplace=inplace)
        assert isinstance(pipe, PlannedPipeline)
        return pipe

    def is_frozen_trainable(self) -> bool:
        return all(step.is_frozen_trainable() for step in self.steps_list())

    def is_frozen_trained(self) -> bool:
        return all(step.is_frozen_trained() for step in self.steps_list())


TrainableOpType_co = TypeVar(
    "TrainableOpType_co", bound=TrainableIndividualOp, covariant=True  # type: ignore
)


class TrainablePipeline(PlannedPipeline[TrainableOpType_co], TrainableOperator):
    def __init__(
        self,
        steps: List[TrainableOpType_co],
        edges: Optional[Iterable[Tuple[TrainableOpType_co, TrainableOpType_co]]] = None,
        _lale_preds: Optional[Dict[int, List[int]]] = None,
        ordered: bool = False,
        _lale_trained=False,
    ) -> None:
        super().__init__(steps, edges=edges, _lale_preds=_lale_preds, ordered=ordered)

    def remove_last(
        self, inplace: bool = False
    ) -> "TrainablePipeline[TrainableOpType_co]":
        pipe = super().remove_last(inplace=inplace)
        assert isinstance(pipe, TrainablePipeline)
        return pipe

    def fit(
        self, X: Any, y: Any = None, **fit_params
    ) -> "TrainedPipeline[TrainedIndividualOp]":
        # filtered_fit_params = _fixup_hyperparams_dict(fit_params)
        X = add_schema(X)
        y = add_schema(y)
        self.validate_schema(X, y)
        trained_steps: List[TrainedIndividualOp] = []
        outputs: Dict[Operator, Tuple[Any, Any]] = {}
        meta_outputs: Dict[Operator, Any] = {}
        edges: List[Tuple[TrainableOpType_co, TrainableOpType_co]] = self.edges()
        trained_map: Dict[TrainableOpType_co, TrainedIndividualOp] = {}

        sink_nodes = self._find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [(X, y)]
                meta_data_inputs: Dict[Operator, Any] = {}
            else:
                inputs = [outputs[pred] for pred in preds]
                # we create meta_data_inputs as a dictionary with metadata from all previous steps
                # Note that if multiple previous steps generate the same key, it will retain only one of those.

                meta_data_inputs = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
            trainable = operator
            assert isinstance(inputs, list) and len(inputs) >= 1
            if len(inputs) == 1:
                input_X, input_y = inputs[0]
            else:
                input_X = [iX for iX, _ in inputs]
                input_y = next(iy for _, iy in inputs)
            if operator.has_method("set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output: Dict[Operator, Any] = {}
            trained: TrainedOperator
            if trainable.is_supervised():
                trained = trainable.fit(input_X, input_y)
            else:
                trained = trainable.fit(input_X)
            trained_map[operator] = trained
            trained_steps.append(trained)
            if (
                trainable not in sink_nodes
            ):  # There is no need to transform/predict on the last node during fit
                if trained.is_transformer():
                    if trained.has_method("transform_X_y"):
                        output = trained.transform_X_y(input_X, input_y)
                    else:
                        output = trained.transform(input_X), input_y
                    if trained.has_method("get_transform_meta_output"):
                        meta_output = (
                            trained._impl_instance().get_transform_meta_output()
                        )
                else:
                    # This is ok because trainable pipelines steps
                    # must only be individual operators
                    if trained.has_method("predict_proba"):  # type: ignore
                        output = trained.predict_proba(input_X), input_y
                    elif trained.has_method("decision_function"):  # type: ignore
                        output = trained.decision_function(input_X), input_y
                    else:
                        output = trained._predict(input_X), input_y
                    if trained.has_method("get_predict_meta_output"):
                        meta_output = trained._impl_instance().get_predict_meta_output()
                outputs[operator] = output
                meta_output_so_far = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
                meta_output_so_far.update(
                    meta_output
                )  # So newest gets preference in case of collisions
                meta_outputs[operator] = meta_output_so_far

        trained_edges = [(trained_map[a], trained_map[b]) for a, b in edges]

        result: TrainedPipeline[TrainedIndividualOp] = TrainedPipeline(
            trained_steps, trained_edges, ordered=True, _lale_trained=True
        )
        self._trained = result
        return result

    @available_if(_final_trained_impl_has("transform"))
    def transform(self, X: Any, y=None) -> Any:
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("transform"), DeprecationWarning)
        try:
            return self._trained.transform(X, y=y)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `transform`.") from exc

    @available_if(_final_trained_impl_has("predict"))
    def predict(self, X, **predict_params) -> Any:
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict"), DeprecationWarning)
        try:
            return self._trained.predict(X, **predict_params)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict`.") from exc

    @available_if(_final_trained_impl_has("predict_proba"))
    def predict_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict_proba"), DeprecationWarning)
        try:
            return self._trained.predict_proba(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict_proba`.") from exc

    @available_if(_final_trained_impl_has("decision_function"))
    def decision_function(self, X):
        """
        .. deprecated:: 0.0.0
           The `decision_function` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `decision_function`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("decision_function"), DeprecationWarning)
        try:
            return self._trained.decision_function(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `decision_function`.") from exc

    @available_if(_final_trained_impl_has("score"))
    def score(self, X, y, **score_params):
        """
        .. deprecated:: 0.0.0
           The `score` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `score`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("score"), DeprecationWarning)
        try:
            return self._trained.score(X, y, **score_params)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `score`.") from exc

    @available_if(_final_trained_impl_has("score_samples"))
    def score_samples(self, X=None):
        """
        .. deprecated:: 0.0.0
           The `score_samples` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `score_samples`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("score_samples"), DeprecationWarning)
        try:
            return self._trained.score_samples(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `score_samples`.") from exc

    @available_if(_final_trained_impl_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_log_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_log_proba`
           on the trained operator returned by `fit` instead.

        """
        warnings.warn(_mutation_warning("predict_log_proba"), DeprecationWarning)
        try:
            return self._trained.predict_log_proba(X)
        except AttributeError as exc:
            raise ValueError("Must call `fit` before `predict_log_proba`.") from exc

    def freeze_trainable(self) -> "TrainablePipeline":
        frozen_steps: List[TrainableOperator] = []
        frozen_map: Dict[Operator, Operator] = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trainable()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = cast(
            TrainablePipeline,
            make_pipeline_graph(frozen_steps, frozen_edges, ordered=True),
        )
        assert result.is_frozen_trainable()
        return result

    def is_transformer(self) -> bool:
        """Checks if the operator is a transformer"""
        sink_nodes = self._find_sink_nodes()
        all_transformers = [
            bool(operator.has_method("transform")) for operator in sink_nodes
        ]
        return all(all_transformers)

    def convert_to_trained(self) -> "TrainedPipeline[TrainedIndividualOp]":
        trained_steps: List[TrainedIndividualOp] = []
        trained_map: Dict[TrainableOpType_co, TrainedIndividualOp] = {}
        for step in self.steps_list():
            trained_step = step.convert_to_trained()
            trained_steps.append(trained_step)
            trained_map[step] = trained_step

        trained_edges = [(trained_map[x], trained_map[y]) for (x, y) in self.edges()]
        return TrainedPipeline(trained_steps, trained_edges, _lale_trained=True)

    def partial_fit(
        self,
        X: Any,
        y: Any = None,
        freeze_trained_prefix: bool = True,
        unsafe: bool = False,
        **fit_params,
    ) -> "TrainedPipeline[TrainedIndividualOp]":
        """partial_fit for a pipeline.
        This method assumes that all but the last node of a pipeline are frozen_trained and
        only the last node needs to be fit using its partial_fit method.
        If that is not the case, and `freeze_trained_prefix` is True, it freezes the prefix
        of the pipeline except the last node if they are trained.

        Parameters
        ----------
        X :
            Features; see partial_fit schema of the last node.
        y:
            Labels/target
        freeze_trained_prefix:
            If True, all but the last node are freeze_trained and only
            the last node is partial_fit.
        unsafe:
            boolean.
            This flag allows users to override the validation that throws an error when the
            the operators in the prefix of this pipeline are not tagged with `has_partial_transform`.
            Setting unsafe to True would perform the transform as if it was row-wise even in the case it may not be.
        fit_params:
            dict
            Additional keyword arguments to be passed to partial_fit of the estimator

        Returns
        -------
        TrainedPipeline :
            A partially trained pipeline, which can be trained further by other calls to partial_fit
        Raises
        ------
        ValueError
            The piepline has a non-frozen prefix
        """
        estimator_only = True

        for operator in self._steps[:-1]:
            if not operator.is_frozen_trained():
                estimator_only = False
        if not estimator_only and not freeze_trained_prefix:
            raise ValueError(
                """partial_fit is only supported on pipelines when all but the last node are frozen_trained and
            only the last node needs to be fit using its partial_fit method. The parameter `freeze_trained_prefix`
            can be set to True if the prefix is trained and needs to be frozen during partial_fit."""
            )
        if hasattr(self, "_trained"):
            # This is the case where partial_fit has been called before,
            # so the partially fit pipeline is stored in _trained.
            # update that object
            self._trained = self._trained.partial_fit(X, y, **fit_params)
            return self._trained
        else:
            # if this is the first time partial_fit is called on this pipeline,
            # we would not have a _trained obj, so convert the prefix to a trained pipeline
            # explicitly and do a transform and partial_fit as expected.
            sink_node = self._steps[-1]
            pipeline_prefix = self.remove_last()
            if not estimator_only and freeze_trained_prefix:
                pipeline_prefix = pipeline_prefix.freeze_trained()
            trained_pipeline_prefix = pipeline_prefix.convert_to_trained()

            transformed_output = trained_pipeline_prefix.transform(X, y)
            if isinstance(transformed_output, tuple):
                transformed_X, transformed_y = transformed_output
            else:
                transformed_X = transformed_output
                transformed_y = y

            trained_sink_node = sink_node.partial_fit(
                transformed_X, transformed_y, **fit_params
            )
            new_pipeline = trained_pipeline_prefix >> trained_sink_node
            self._trained = new_pipeline
            return new_pipeline

    def freeze_trained(self) -> "TrainedPipeline":
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trained()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = TrainedPipeline(
            frozen_steps, frozen_edges, ordered=True, _lale_trained=True
        )
        assert result.is_frozen_trained()
        return result


TrainedOpType_co = TypeVar("TrainedOpType_co", bound=TrainedIndividualOp, covariant=True)  # type: ignore


class TrainedPipeline(TrainablePipeline[TrainedOpType_co], TrainedOperator):
    def __new__(cls, *args, _lale_trained=False, **kwargs):
        if "steps" not in kwargs or _lale_trained:
            obj = super().__new__(TrainedPipeline)  # type: ignore
            return obj
        else:
            # unless _lale_trained=True, we actually want to return a Trainable
            obj = super().__new__(TrainablePipeline)  # type: ignore
            # apparently python does not call __ini__ if the type returned is not the
            # expected type
            obj.__init__(*args, **kwargs)
            return obj

    def __init__(
        self,
        steps: List[TrainedOpType_co],
        edges: Optional[List[Tuple[TrainedOpType_co, TrainedOpType_co]]] = None,
        _lale_preds: Optional[Dict[int, List[int]]] = None,
        ordered: bool = False,
        _lale_trained=False,
    ) -> None:
        super().__init__(steps, edges=edges, _lale_preds=_lale_preds, ordered=ordered)

    def remove_last(self, inplace: bool = False) -> "TrainedPipeline[TrainedOpType_co]":
        pipe = super().remove_last(inplace)
        assert isinstance(pipe, TrainedPipeline)
        return pipe

    def _predict(self, X: Any, y: Any = None, **predict_params):
        return self._predict_based_on_type(
            "predict", "_predict", X, y, **predict_params
        )

    def predict(self, X, **predict_params) -> Any:
        result = self._predict(X, **predict_params)
        if isinstance(result, NDArrayWithSchema):
            return strip_schema(result)  # otherwise scorers return zero-dim array
        return result

    @available_if(_final_impl_has("transform"))
    def transform(self, X: Any, y: Any = None) -> Any:
        # TODO: What does a transform on a pipeline mean, if the last step is not a transformer
        # can it be just the output of predict of the last step?
        # If this implementation changes, check to make sure that the implementation of
        # self.is_transformer is kept in sync with the new assumptions.
        return self._predict_based_on_type("transform", "transform", X, y)

    @available_if(_final_impl_has("transform_X_y"))
    def transform_X_y(self, X: Any, y: Any = None) -> Any:
        return self._predict_based_on_type("transform_X_y", "transform_X_y", X, y)

    def _predict_based_on_type(
        self, impl_method_name, operator_method_name, X=None, y=None, **kwargs
    ):
        outputs = {}
        meta_outputs = {}
        sink_nodes = self._find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [(X, y)]
                meta_data_inputs = {}
            else:
                inputs = [outputs[pred] for pred in preds]
                # we create meta_data_inputs as a dictionary with metadata from all previous steps
                # Note that if multiple previous steps generate the same key, it will retain only one of those.

                meta_data_inputs = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
            assert isinstance(inputs, list) and len(inputs) >= 1
            if len(inputs) == 1:
                input_X, input_y = inputs[0]
            else:
                input_X = [iX for iX, _ in inputs]
                input_y = next(iy for _, iy in inputs)
            if operator.has_method("set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output = {}
            if operator in sink_nodes:
                if operator.has_method(
                    impl_method_name
                ):  # Since this is pipeline's predict, we should invoke predict from sink nodes
                    method_to_call_on_operator = getattr(operator, operator_method_name)
                    if operator_method_name == "score":
                        output = (
                            method_to_call_on_operator(input_X, input_y, **kwargs),
                            input_y,
                        )
                    elif operator_method_name == "transform_X_y":
                        output = method_to_call_on_operator(input_X, input_y, **kwargs)
                    else:
                        output = method_to_call_on_operator(input_X, **kwargs), input_y
                else:
                    raise AttributeError(
                        f"The sink node {type(operator.impl)} of the pipeline does not support {operator_method_name}"
                    )
            elif operator.is_transformer():
                if operator.has_method("transform_X_y"):
                    output = operator.transform_X_y(input_X, input_y)
                else:
                    output = operator.transform(input_X), input_y
                if hasattr(operator._impl, "get_transform_meta_output"):
                    meta_output = operator._impl_instance().get_transform_meta_output()
            elif operator.has_method(
                "predict_proba"
            ):  # For estimator as a transformer, use predict_proba if available
                output = operator.predict_proba(input_X), input_y
            elif operator.has_method(
                "decision_function"
            ):  # For estimator as a transformer, use decision_function if available
                output = operator.decision_function(input_X), input_y
            else:
                output = operator._predict(input_X), input_y
                if operator.has_method("get_predict_meta_output"):
                    meta_output = operator._impl_instance().get_predict_meta_output()
            outputs[operator] = output
            meta_output_so_far = {
                key: meta_outputs[pred][key]
                for pred in preds
                if meta_outputs[pred] is not None
                for key in meta_outputs[pred]
            }
            meta_output_so_far.update(
                meta_output
            )  # So newest gets preference in case of collisions
            meta_outputs[operator] = meta_output_so_far
        result_X, result_y = outputs[self._steps[-1]]
        if operator_method_name == "transform_X_y":
            return result_X, result_y
        return result_X

    @available_if(_final_impl_has("predict_proba"))
    def predict_proba(self, X: Any):
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        return self._predict_based_on_type("predict_proba", "predict_proba", X)

    @available_if(_final_impl_has("decision_function"))
    def decision_function(self, X: Any):
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        return self._predict_based_on_type("decision_function", "decision_function", X)

    @available_if(_final_impl_has("score"))
    def score(self, X: Any, y: Any, **score_params):
        """Performance evaluation with a default metric based on the final estimator.

        Parameters
        ----------
        X :
            Features.
        y:
            Ground truth labels.
        score_params:
            Any additional parameters expected by the score function of
            the final estimator. These will be ignored for now.
        Returns
        -------
        score :
            Performance metric value.
        """
        return self._predict_based_on_type("score", "score", X, y)

    @available_if(_final_impl_has("score_samples"))
    def score_samples(self, X: Any = None):
        """Scores for each sample in X. There type of scores is based on the last operator in the pipeline.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            Scores per sample.
        """
        return self._predict_based_on_type("score_samples", "score_samples", X)

    @available_if(_final_impl_has("predict_log_proba"))
    def predict_log_proba(self, X: Any):
        """Predicted class log-probabilities for X.

        Parameters
        ----------
        X :
            Features.

        Returns
        -------
        result :
            Class log probabilities.
        """
        return self._predict_based_on_type("predict_log_proba", "predict_log_proba", X)

    def transform_with_batches(self, X: Any, y: Any = None, serialize: bool = True):
        """[summary]

        Parameters
        ----------
        X : Any
            [description]
        y : [type], optional
            by default None
        serialize: boolean
            should data be serialized if needed
        Returns
        -------
        [type]
            [description]
        """
        outputs: Dict[TrainedOpType_co, tuple] = {}
        serialization_out_dir: str = ""
        if serialize:
            serialization_out_dir = os.path.join(
                os.path.dirname(__file__), "temp_serialized"
            )
            if not os.path.exists(serialization_out_dir):
                os.mkdir(serialization_out_dir)

        sink_nodes = self._find_sink_nodes()
        sink_node = sink_nodes[0]
        operator_idx = 0
        inputs: Any
        output = None

        for batch_data in X:  # batching_transformer will output only one obj
            if isinstance(batch_data, tuple):
                batch_X, batch_y = batch_data
            else:
                batch_X = batch_data
                batch_y = None

            for operator in self._steps:
                preds = self._preds[operator]
                if len(preds) == 0:
                    inputs = batch_X
                else:
                    inputs = [
                        (
                            outputs[pred][0]
                            if isinstance(outputs[pred], tuple)
                            else outputs[pred]
                        )
                        for pred in preds
                    ]
                if len(inputs) == 1:
                    inputs = inputs[0]
                trained = operator
                if trained.is_transformer():
                    assert not trained.has_method("transform_X_y"), "TODO"
                    batch_output = trained.transform(inputs, batch_y)
                else:
                    if trained in sink_nodes:
                        batch_output = trained._predict(
                            X=inputs
                        )  # We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if trained.has_method("predict_proba"):  # type: ignore
                            batch_output = trained.predict_proba(X=inputs)
                        elif trained.has_method("decision_function"):  # type: ignore
                            batch_output = trained.decision_function(X=inputs)
                        else:
                            batch_output = trained._predict(X=inputs)
                if trained == sink_node:
                    if isinstance(batch_output, tuple):
                        output = append_batch(
                            output, (batch_output[0], batch_output[1])
                        )
                    else:
                        output = append_batch(output, batch_output)
                outputs[operator] = batch_output
                operator_idx += 1

            #     if serialize:
            #         output = lale.helpers.write_batch_output_to_file(
            #             output,
            #             os.path.join(
            #                 serialization_out_dir,
            #                 "fit_with_batches" + str(operator_idx) + ".hdf5",
            #             ),
            #             len(inputs.dataset),
            #             batch_idx,
            #             batch_X,
            #             batch_y,
            #             batch_out_X,
            #             batch_out_y,
            #         )
            #     else:
            #         if batch_out_y is not None:
            #             output = lale.helpers.append_batch(
            #                 output, (batch_output, batch_out_y)
            #             )
            #         else:
            #             output = lale.helpers.append_batch(output, batch_output)
            # if serialize:
            #     output.close()  # type: ignore
            #     output = lale.helpers.create_data_loader(
            #         os.path.join(
            #             serialization_out_dir,
            #             "fit_with_batches" + str(operator_idx) + ".hdf5",
            #         ),
            #         batch_size=inputs.batch_size,
            #     )
            # else:
            #     if isinstance(output, tuple):
            #         output = lale.helpers.create_data_loader(
            #             X=output[0], y=output[1], batch_size=inputs.batch_size
            #         )
            #     else:
            #         output = lale.helpers.create_data_loader(
            #             X=output, y=None, batch_size=inputs.batch_size
            #         )
            # outputs[operator] = output
            # operator_idx += 1

        return_data = output  # outputs[self._steps[-1]]#.dataset.get_data()
        # if serialize:
        #     shutil.rmtree(serialization_out_dir)

        return return_data

    def freeze_trainable(self) -> "TrainedPipeline":
        result = super().freeze_trainable()
        return cast(TrainedPipeline, result)

    def partial_fit(
        self,
        X: Any,
        y: Any = None,
        freeze_trained_prefix: bool = True,
        unsafe: bool = False,
        classes: Any = None,
        **fit_params,
    ) -> "TrainedPipeline[TrainedIndividualOp]":
        """partial_fit for a pipeline.
        This method assumes that all but the last node of a pipeline are frozen_trained and
        only the last node needs to be fit using its partial_fit method.
        If that is not the case, and `freeze_trained_prefix` is True, it freezes the prefix
        of the pipeline except the last node if they are trained.

        Parameters
        ----------
        X :
            Features; see partial_fit schema of the last node.
        y:
            Labels/target
        freeze_trained_prefix:
            If True, all but the last node are freeze_trained and only
            the last node is partial_fit.
        unsafe:
            boolean.
            This flag allows users to override the validation that throws an error when the
            the operators in the prefix of this pipeline are not tagged with `has_partial_transform`.
            Setting unsafe to True would perform the transform as if it was row-wise even in the case it may not be.
        fit_params:
            dict
            Additional keyword arguments to be passed to partial_fit of the estimator
        classes: Any

        Returns
        -------
        TrainedPipeline :
            A partially trained pipeline, which can be trained further by other calls to partial_fit


        Raises
        ------
        ValueError
            The piepline has a non-frozen prefix
        """
        estimator_only = True

        for operator in self._steps[:-1]:
            if not operator.is_frozen_trained():
                estimator_only = False
        if not estimator_only and not freeze_trained_prefix:
            raise ValueError(
                """partial_fit is only supported on pipelines when all but the last node are frozen_trained and
            only the last node needs to be fit using its partial_fit method. The parameter `freeze_trained_prefix`
            can be set to True if the prefix is trained and needs to be frozen during partial_fit."""
            )
        sink_node = self._steps[-1]
        pipeline_prefix = self.remove_last()
        if not estimator_only and freeze_trained_prefix:
            pipeline_prefix = pipeline_prefix.freeze_trained()
        transformed_output = pipeline_prefix.transform(X, y)
        if isinstance(transformed_output, tuple):
            transformed_X, transformed_y = transformed_output
        else:
            transformed_X = transformed_output
            transformed_y = y
        try:
            trained_sink_node = sink_node.partial_fit(
                transformed_X, transformed_y, classes=classes, **fit_params
            )
        except TypeError:  # occurs when `classes` is not expected
            trained_sink_node = sink_node.partial_fit(
                transformed_X, transformed_y, **fit_params
            )
        trained_pipeline = pipeline_prefix >> trained_sink_node
        return trained_pipeline


OperatorChoiceType_co = TypeVar("OperatorChoiceType_co", bound=Operator, covariant=True)


class OperatorChoice(PlannedOperator, Generic[OperatorChoiceType_co]):
    _name: str
    _steps: List[OperatorChoiceType_co]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["steps"] = self._steps
        out["name"] = self._name

        indices: Dict[str, int] = {}

        def make_indexed(name: str) -> str:
            idx = 0
            if name in indices:
                idx = indices[name] + 1
                indices[name] = idx
            else:
                indices[name] = 0
            return make_indexed_name(name, idx)

        if deep:
            for op in self._steps:
                name = make_indexed(op.name())
                nested_params = op.get_params(deep=deep)
                if nested_params:
                    out.update(nest_HPparams(name, nested_params))
        return out

    def set_params(self, **impl_params):
        """This implements the set_params, as per the scikit-learn convention,
        extended as documented in the module docstring"""
        return self._with_params(True, **impl_params)

    # TODO: enhance to support setting params of a choice without picking a choice
    # TODO: also, enhance to support mutating it in place?
    def _with_params(self, try_mutate: bool, **impl_params) -> Operator:
        """
        This method updates the parameters of the operator.
        If try_mutate is set, it will attempt to update the operator in place
        this may not always be possible
        """
        choices = self.steps_list()
        choice_index: int
        chosen_params: Dict[str, Any]
        if len(choices) == 1:
            choice_index = 0
            chosen_params = impl_params
        else:
            (choice_index, chosen_params) = partition_sklearn_choice_params(impl_params)

        assert 0 <= choice_index < len(choices)
        choice: Operator = choices[choice_index]

        new_step = choice._with_params(try_mutate, **chosen_params)
        # in the functional case
        # we remove the OperatorChoice, replacing it with the branch that was taken
        # TODO: in the mutating case, we could update this choice
        return new_step

    def __init__(self, steps, name: Optional[str] = None) -> None:
        if name is None or name == "":
            name = assignee_name(level=2)
        if name is None or name == "":
            name = "OperatorChoice"

        self._name = name
        self._steps = steps

    def steps_list(self) -> List[OperatorChoiceType_co]:
        return self._steps

    @property
    def steps(self) -> List[Tuple[str, OperatorChoiceType_co]]:
        """This is meant to function similarly to the scikit-learn steps property
        and for linear pipelines, should behave the same
        """
        return [(s.name(), s) for s in self._steps]

    def fit(self, X: Any, y: Any = None, **fit_params):
        if len(self.steps_list()) == 1:
            s = self.steps_list()[0]
            if s is not None:
                f = getattr(s, "fit", None)
                if f is not None:
                    return f(X, y, **fit_params)
                else:
                    return None
            else:
                return None
        else:
            # This call is to get the correct error message
            # calling getattr(self, "fit") would result in
            # infinite recursion, but this explicit call works
            return self.__getattr__("fit")  # pylint:disable=unnecessary-dunder-call

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator imnplementations are compatible"""
        if not isinstance(other, OperatorChoice):
            return False
        my_steps = self.steps_list()
        other_steps = other.steps_list()
        if len(my_steps) != len(other_steps):
            return False

        for m, o in zip(my_steps, other_steps):
            if not m._has_same_impl(o):
                return False
        return True

    def is_supervised(self) -> bool:
        s = self.steps_list()
        if len(s) == 0:
            return False
        return self.steps_list()[-1].is_supervised()

    def validate_schema(self, X: Any, y: Any = None):
        for step in self.steps_list():
            step.validate_schema(X, y)

    def transform_schema(self, s_X: JSON_TYPE):
        from lale.settings import disable_data_schema_validation

        if disable_data_schema_validation:
            return {}
        else:
            transformed_schemas = [st.transform_schema(s_X) for st in self.steps_list()]
            result = join_schemas(*transformed_schemas)
            return result

    def input_schema_fit(self) -> JSON_TYPE:
        pipeline_inputs = [s.input_schema_fit() for s in self.steps_list()]
        result = join_schemas(*pipeline_inputs)
        return result

    def is_frozen_trainable(self) -> bool:
        return all(step.is_frozen_trainable() for step in self.steps_list())

    def is_classifier(self) -> bool:
        for op in self.steps_list():
            if not op.is_classifier():
                return False
        return True

    def get_defaults(self) -> Mapping[str, Any]:
        defaults_list: Iterable[Mapping[str, Any]] = (
            s.get_defaults() for s in self.steps_list()
        )

        defaults: Dict[str, Any] = {}
        for d in defaults_list:
            defaults.update(d)

        return defaults


class _PipelineFactory:
    def __init__(self):
        pass

    def __call__(self, steps: List[Any]):
        warnings.warn(
            "lale.operators.Pipeline is deprecated, use sklearn.pipeline.Pipeline or lale.lib.sklearn.Pipeline instead",
            DeprecationWarning,
        )
        for i in range(len(steps)):  # pylint:disable=consider-using-enumerate
            op = steps[i]
            if isinstance(op, tuple):
                assert isinstance(op[1], Operator)
                op[1]._set_name(op[0])
                steps[i] = op[1]
        return make_pipeline(*steps)


Pipeline = _PipelineFactory()


def _pipeline_graph_class(steps) -> Type[PlannedPipeline]:
    isTrainable: bool = True
    isTrained: bool = True
    for operator in steps:
        if not isinstance(operator, TrainedOperator):
            isTrained = False  # Even if a single step is not trained, the pipeline can't be used for predict/transform
            # without training it first
        if isinstance(operator, OperatorChoice) or not isinstance(
            operator, TrainableOperator
        ):
            isTrainable = False
    if isTrained:
        return TrainedPipeline
    elif isTrainable:
        return TrainablePipeline
    else:
        return PlannedPipeline


@overload
def make_pipeline_graph(
    steps: List[TrainedOperator],
    edges: List[Tuple[Operator, Operator]],
    ordered: bool = False,
) -> TrainedPipeline: ...


@overload
def make_pipeline_graph(
    steps: List[TrainableOperator],
    edges: List[Tuple[Operator, Operator]],
    ordered: bool = False,
) -> TrainablePipeline: ...


@overload
def make_pipeline_graph(
    steps: List[Operator],
    edges: List[Tuple[Operator, Operator]],
    ordered: bool = False,
) -> PlannedPipeline: ...


def make_pipeline_graph(steps, edges, ordered=False) -> PlannedPipeline:
    """
    Based on the state of the steps, it is important to decide an appropriate type for
    a new Pipeline. This method will decide the type, create a new Pipeline of that type and return it.
    #TODO: If multiple independently trained components are composed together in a pipeline,
    should it be of type TrainedPipeline?
    Currently, it will be TrainablePipeline, i.e. it will be forced to train it again.
    """
    pipeline_class = _pipeline_graph_class(steps)
    if pipeline_class is TrainedPipeline:
        return TrainedPipeline(steps, edges, ordered=ordered, _lale_trained=True)
    else:
        return pipeline_class(steps, edges, ordered=ordered)


@overload
def make_pipeline(*orig_steps: TrainedOperator) -> TrainedPipeline: ...


@overload
def make_pipeline(*orig_steps: TrainableOperator) -> TrainablePipeline: ...


@overload
def make_pipeline(*orig_steps: Union[Operator, Any]) -> PlannedPipeline: ...


def make_pipeline(*orig_steps):
    steps: List[Operator] = []
    edges: List[Tuple[Operator, Operator]] = []
    prev_op: Optional[Operator] = None
    for curr_op in orig_steps:
        if isinstance(prev_op, BasePipeline):
            prev_leaves: List[Operator] = prev_op._find_sink_nodes()
        else:
            prev_leaves = [] if prev_op is None else [prev_op]
        if isinstance(curr_op, BasePipeline):
            curr_roots: List[Operator] = curr_op._find_source_nodes()
            steps.extend(curr_op.steps_list())
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name=curr_op.__class__.__name__)
            curr_roots = [curr_op]
            steps.append(curr_op)
        edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])
        prev_op = curr_op
    return make_pipeline_graph(steps, edges, ordered=True)


@overload
def make_union_no_concat(*orig_steps: TrainedOperator) -> TrainedPipeline: ...


@overload
def make_union_no_concat(*orig_steps: TrainableOperator) -> TrainablePipeline: ...


@overload
def make_union_no_concat(*orig_steps: Union[Operator, Any]) -> PlannedPipeline: ...


def make_union_no_concat(*orig_steps):  # type: ignore
    steps, edges = [], []
    for curr_op in orig_steps:
        if isinstance(curr_op, BasePipeline):
            steps.extend(curr_op._steps)
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name=curr_op.__class__.__name__)
            steps.append(curr_op)
    return make_pipeline_graph(steps, edges, ordered=True)


@overload
def make_union(*orig_steps: TrainedOperator) -> TrainedPipeline: ...


@overload
def make_union(*orig_steps: TrainableOperator) -> TrainablePipeline: ...


@overload
def make_union(*orig_steps: Union[Operator, Any]) -> PlannedPipeline: ...


def make_union(*orig_steps):  # type: ignore
    from lale.lib.rasl import ConcatFeatures

    return make_union_no_concat(*orig_steps) >> ConcatFeatures()


def make_choice(
    *orig_steps: Union[Operator, Any], name: Optional[str] = None
) -> OperatorChoice:
    if name is None:
        name = ""
    name_: str = name  # to make mypy happy
    steps: List[Operator] = []
    for operator in orig_steps:
        if isinstance(operator, OperatorChoice):
            steps.extend(operator.steps_list())
        else:
            if not isinstance(operator, Operator):
                operator = make_operator(operator, name=operator.__class__.__name__)
            steps.append(operator)
        name_ = name_ + " | " + operator.name()
    return OperatorChoice(steps, name_[3:])


def _fixup_hyperparams_dict(d):
    d1 = remove_defaults_dict(d)
    d2 = {k: val_wrapper.unwrap(v) for k, v in d1.items()}
    return d2


CustomizeOpType = TypeVar("CustomizeOpType", bound=PlannedIndividualOp)


def customize_schema(  # pylint: disable=differing-param-doc,differing-type-doc
    op: CustomizeOpType,
    schemas: Optional[Schema] = None,
    relevantToOptimizer: Optional[List[str]] = None,
    constraint: Union[Schema, JSON_TYPE, List[Union[Schema, JSON_TYPE]], None] = None,
    tags: Optional[Dict] = None,
    forwards: Union[bool, List[str], None] = None,
    set_as_available: bool = False,
    **kwargs: Union[Schema, JSON_TYPE, None],
) -> CustomizeOpType:
    """Return a new operator with a customized schema

    Parameters
    ----------
    op: Operator
        The base operator to customize
    schemas : Schema
        A dictionary of json schemas for the operator. Override the entire schema and ignore other arguments
    input : Schema
        (or `input_*`) override the input schema for method `*`.
        `input_*` must be an existing method (already defined in the schema for lale operators, existing method for external operators)
    output : Schema
        (or `output_*`) override the output schema for method `*`.
        `output_*` must be an existing method (already defined in the schema for lale operators, existing method for external operators)
    relevantToOptimizer : String list
        update the set parameters that will be optimized.
    constraint : Schema
        Add a constraint in JSON schema format.
    tags : Dict
        Override the tags of the operator.
    forwards: boolean or a list of strings
        Which methods/properties to forward to the underlying impl.  (False for none, True for all).
    set_as_available: bool
        Override the list of available operators so `get_available_operators` returns this customized operator.
    kwargs : Schema
        Override the schema of the hyperparameter.
        `param` must be an existing parameter (already defined in the schema for lale operators, __init__ parameter for external operators)

    Returns
    -------
    PlannedIndividualOp
        Copy of the operator with a customized schema
    """
    op_index = -1
    try:
        op_index = _all_available_operators.index(op)
    except ValueError:
        pass
    # TODO: why are we doing a deeopcopy here?
    op = copy.deepcopy(op)
    methods = ["fit", "transform", "predict", "predict_proba", "decision_function"]
    # explicitly enable the hyperparams schema check because it is important
    from lale.settings import (
        disable_hyperparams_schema_validation,
        set_disable_hyperparams_schema_validation,
    )

    existing_disable_hyperparams_schema_validation = (
        disable_hyperparams_schema_validation
    )
    set_disable_hyperparams_schema_validation(False)

    if schemas is not None:
        schemas.schema["$schema"] = "http://json-schema.org/draft-04/schema#"
        validate_is_schema(schemas.schema)
        op._schemas = schemas.schema
    else:
        if relevantToOptimizer is not None:
            assert isinstance(relevantToOptimizer, list)
            op._schemas["properties"]["hyperparams"]["allOf"][0][
                "relevantToOptimizer"
            ] = relevantToOptimizer
        if constraint is not None:
            cl: List[Union[Schema, JSON_TYPE]]
            if isinstance(constraint, list):
                cl = constraint
            else:
                cl = [constraint]

            for c in cl:
                if isinstance(c, Schema):
                    c = c.schema
                op._schemas["properties"]["hyperparams"]["allOf"].append(c)
        if tags is not None:
            assert isinstance(tags, dict)
            op._schemas["tags"] = tags
        if forwards is not None:
            assert isinstance(forwards, (bool, list))
            op._schemas["forwards"] = forwards

        for arg, value in kwargs.items():
            if value is not None and isinstance(value, Schema):
                value = value.schema
            if value is not None:
                validate_is_schema(value)
            if arg in [p + n for p in ["input_", "output_"] for n in methods]:
                # multiple input types (e.g., fit, predict)
                assert value is not None
                validate_method(op, arg)
                op._schemas["properties"][arg] = value
            elif value is None:
                scm = op._schemas["properties"]["hyperparams"]["allOf"][0]
                if "required" in scm:
                    scm["required"] = [k for k in scm["required"] if k != arg]
                if "relevantToOptimizer" in scm:
                    scm["relevantToOptimizer"] = [
                        k for k in scm["relevantToOptimizer"] if k != arg
                    ]
                if "properties" in scm:
                    scm["properties"] = {
                        k: scm["properties"][k] for k in scm["properties"] if k != arg
                    }
            else:
                op._schemas["properties"]["hyperparams"]["allOf"][0]["properties"][
                    arg
                ] = value
    # since the schema has changed, we need to invalidate any
    # cached enum attributes
    op._invalidate_enum_attributes()
    set_disable_hyperparams_schema_validation(
        existing_disable_hyperparams_schema_validation
    )
    # we also need to prune the hyperparameter, if any, removing defaults (which may have changed)
    op._hyperparams = op.hyperparams()
    if set_as_available and op_index >= 0:
        _all_available_operators[op_index] = op
    return op


CloneOpType = TypeVar("CloneOpType", bound=Operator)


def clone_op(op: CloneOpType, name: Optional[str] = None) -> CloneOpType:
    """Clone any operator."""
    nop = clone(op)
    if name:
        nop._set_name(name)
    return nop


def with_structured_params(
    try_mutate: bool, k, params: Dict[str, Any], hyper_parent
) -> None:
    # need to handle the different encoding schemes used
    if params is None:
        return
    if structure_type_name in params:
        # this is a structured type
        structure_type = params[structure_type_name]
        type_params, sub_params = partition_sklearn_params(params)

        hyper = None
        if isinstance(hyper_parent, dict):
            hyper = hyper_parent.get(k, None)
        elif isinstance(hyper_parent, list) and k < len(hyper_parent):
            hyper = hyper_parent[k]
        if hyper is None:
            hyper = {}
        elif isinstance(hyper, tuple):
            # to make it mutable
            hyper = list(hyper)

        del type_params[structure_type_name]
        actual_key: Union[str, int]
        for elem_key, elem_value in type_params.items():
            if elem_value is not None:
                if not isinstance(hyper, dict):
                    assert is_numeric_structure(structure_type)
                    actual_key = int(elem_key)
                    # we may need to extend the array
                    try:
                        hyper[actual_key] = elem_value
                    except IndexError:
                        assert 0 <= actual_key
                        hyper.extend((actual_key - len(hyper)) * [None])
                        hyper.append(elem_value)
                else:
                    actual_key = elem_key
                    hyper[actual_key] = elem_value

        for elem_key, elem_params in sub_params.items():
            if not isinstance(hyper, dict):
                assert is_numeric_structure(structure_type)
                actual_key = int(elem_key)
            else:
                actual_key = elem_key
            with_structured_params(try_mutate, actual_key, elem_params, hyper)
        if isinstance(hyper, dict) and is_numeric_structure(structure_type):
            max_key = max((int(x) for x in hyper.keys()))
            hyper = [hyper.get(str(x), None) for x in range(max_key)]
        if structure_type == "tuple":
            hyper = tuple(hyper)
        hyper_parent[k] = hyper
    else:
        # if it is not a structured parameter
        # then it must be a nested higher order operator
        sub_op = hyper_parent[k]
        if isinstance(sub_op, list):
            if len(sub_op) == 1:
                sub_op = sub_op[0]
            else:
                (disc, chosen_params) = partition_sklearn_choice_params(params)
                assert 0 <= disc < len(sub_op)
                sub_op = sub_op[disc]
                params = chosen_params
        trainable_sub_op = sub_op._with_params(try_mutate, **params)
        hyper_parent[k] = trainable_sub_op
