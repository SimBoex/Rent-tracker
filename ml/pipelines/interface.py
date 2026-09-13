"""Regressor factory contract for plug-in models in the OMI pipeline."""

from __future__ import annotations

from typing import Protocol

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge


class RegressorFactory(Protocol):
    def make(self, cat_idx: list[int]) -> BaseEstimator: ...


class HistGradientBoostingFactory:
    """Default baseline: HGB with native categorical indices after ColumnTransformer."""

    def make(self, cat_idx: list[int]) -> BaseEstimator:
        return HistGradientBoostingRegressor(
            random_state=42,
            categorical_features=cat_idx,  # pyright: ignore[reportArgumentType]
        )


class RidgeFactory:
    """Simple linear baseline; ignores cat_idx (cats already ordinal-encoded)."""

    def make(self, cat_idx: list[int]) -> BaseEstimator:
        return Ridge()


class LinearRegressionFactory:
    """OLS linear regression; ignores cat_idx (cats already ordinal-encoded)."""

    def make(self, cat_idx: list[int]) -> BaseEstimator:
        return LinearRegression()
