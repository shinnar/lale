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

import autoai_libs.transformers.exportable
import lale.datasets.data_schemas
import lale.docstrings
import lale.operators

class boolean2floatImpl():
    def __init__(self, activate_flag):
        self._hyperparams = {
            'activate_flag': activate_flag}
        self._autoai_tfm = autoai_libs.transformers.exportable.boolean2float(**self._hyperparams)

    def fit(self, X, y=None):
        self._autoai_tfm.fit(X, y)
        return self

    def transform(self, X):
        raw = self._autoai_tfm.transform(X)
        s_X = lale.datasets.data_schemas.to_schema(X)
        s_result = self.transform_schema(s_X)
        result = lale.datasets.data_schemas.add_schema(raw, s_result, recalc=True)
        assert result.json_schema == s_result
        return result

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        if self._hyperparams['activate_flag']:
            result = {
                'type': 'array',
                'items': {'type': 'array', 'items': {'type': 'number'}}}
        else:
            result = s_X
        return result

_hyperparams_schema = {
    'allOf': [{
        'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.',
        'type': 'object',
        'additionalProperties': False,
        'required': ['activate_flag'],
        'relevantToOptimizer': [],
        'properties': {
            'activate_flag': {
                'description': 'If False, transform(X) outputs the input numpy array X unmodified.',
                'type': 'boolean',
                'default': True}}}]}

_input_fit_schema = {
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'type': 'array',
            'items': {'type': 'array', 'items': {'laleType': 'Any'}}},
        'y': {
            'laleType': 'Any'}}}

_input_transform_schema = {
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'type': 'array',
            'items': {'type': 'array', 'items': {'laleType': 'Any'}}}}}

_output_transform_schema = {
    'description': 'Features; the outer array is over samples.',
    'type': 'array',
    'items': {'type': 'array', 'items': {'laleType': 'Any'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': """Operator from `autoai_libs`_. Converts strings that represent booleans to floats and replaces missing values with np.nan.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai.boolean2float.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

lale.docstrings.set_docstrings(boolean2floatImpl, _combined_schemas)

boolean2float = lale.operators.make_operator(boolean2floatImpl, _combined_schemas)