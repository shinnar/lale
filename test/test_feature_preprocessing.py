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

import unittest

import lale
from lale.lib.sklearn import PCA


class TestFeaturePreprocessing(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


def create_function_test_feature_preprocessor(fproc_name):
    def test_feature_preprocessor(self):
        X_train, y_train = self.X_train, self.y_train
        import importlib

        module_name = ".".join(fproc_name.split(".")[0:-1])
        class_name = fproc_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        fproc = class_()

        from lale.lib.sklearn.one_hot_encoder import OneHotEncoderImpl

        if fproc._impl_class() == OneHotEncoderImpl:
            # fproc = OneHotEncoder(handle_unknown = 'ignore')
            # remove the hack when this is fixed
            fproc = PCA()
        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(fproc.input_schema_fit())
        lale.type_checking.validate_is_schema(fproc.input_schema_transform())
        lale.type_checking.validate_is_schema(fproc.output_schema_transform())
        lale.type_checking.validate_is_schema(fproc.hyperparam_schema())

        # test_init_fit_transform
        trained = fproc.fit(self.X_train, self.y_train)
        _ = trained.transform(self.X_test)

        # test_predict_on_trainable
        trained = fproc.fit(X_train, y_train)
        fproc.transform(X_train)

        # test_to_json
        fproc.to_json()

        # test_in_a_pipeline
        # This test assumes that the output of feature processing is compatible with LogisticRegression
        from lale.lib.sklearn import LogisticRegression

        pipeline = fproc >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # Tune the pipeline with LR using Hyperopt
        from lale.lib.lale import Hyperopt

        hyperopt = Hyperopt(estimator=pipeline, max_evals=1, verbose=True, cv=3)
        trained = hyperopt.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    test_feature_preprocessor.__name__ = "test_{0}".format(fproc_name.split(".")[-1])
    return test_feature_preprocessor


feature_preprocessors = [
    "lale.lib.sklearn.PolynomialFeatures",
    "lale.lib.sklearn.PCA",
    "lale.lib.sklearn.Nystroem",
    "lale.lib.sklearn.Normalizer",
    "lale.lib.sklearn.MinMaxScaler",
    "lale.lib.sklearn.OneHotEncoder",
    "lale.lib.sklearn.SimpleImputer",
    "lale.lib.sklearn.StandardScaler",
    "lale.lib.sklearn.FeatureAgglomeration",
    "lale.lib.sklearn.RobustScaler",
    "lale.lib.sklearn.QuantileTransformer",
]
for fproc in feature_preprocessors:
    setattr(
        TestFeaturePreprocessing,
        "test_{0}".format(fproc.split(".")[-1]),
        create_function_test_feature_preprocessor(fproc),
    )
