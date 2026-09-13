"""Build sklearn Pipeline: OMI preprocessing + plug-in regressor."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml.pipelines.interface import HistGradientBoostingFactory, RegressorFactory


class FillNanConstant(BaseEstimator, TransformerMixin):
    """Fill NaN with a constant; never drops columns (unlike older SimpleImputer)."""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(arr, nan=self.fill_value)


class PipelineBuilder:
    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        regressor_factory: RegressorFactory | None = None,
    ):
        self.numeric_features = (
            list(NUMERIC_FEATURES) if numeric_features is None else list(numeric_features)
        )
        self.categorical_features = (
            list(CATEGORICAL_FEATURES)
            if categorical_features is None
            else list(categorical_features)
        )
        self.feature_cols = self.numeric_features + self.categorical_features
        self.regressor_factory: RegressorFactory = (
            HistGradientBoostingFactory()
            if regressor_factory is None
            else regressor_factory
        )

    def build(self) -> Pipeline:
        transformers: list[tuple] = []
        if self.numeric_features:
            # Missing lag (first semester) → 0; Ridge/LR reject NaN.
            transformers.append(
                ("num", FillNanConstant(0.0), self.numeric_features)
            )
        transformers.append(
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                self.categorical_features,
            )
        )
        pre = ColumnTransformer(transformers=transformers)
        # After transform: [numeric..., categorical...]
        n_num = len(self.numeric_features)
        cat_idx = list(range(n_num, n_num + len(self.categorical_features)))
        return Pipeline(
            steps=[
                ("pre", pre),
                ("model", self.regressor_factory.make(cat_idx)),
            ]
        )
