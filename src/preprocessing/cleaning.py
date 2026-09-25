"""
Standardized feature-level data cleaning pipeline.

Combines three steps into one scikit-learn Pipeline so cleaning is always
applied identically to train, test, and future production data:

    1. Missing value imputation (median, by default -- robust to skew)
    2. Outlier handling (IQR-based capping / winsorization)
    3. Feature scaling (RobustScaler by default -- less sensitive to the
       outliers that step 2 didn't fully remove, since capping is a
       compromise, not a guarantee of a clean distribution)

Fit this ONCE on the training split only, then use the same fitted
pipeline to transform validation/test/future data. Never re-fit on new
data at inference time -- that would leak future distribution info
backward, which conflicts with the project's leakage-prevention rules.

Usage:
    from cleaning import build_cleaning_pipeline, save_pipeline, load_pipeline

    pipeline = build_cleaning_pipeline()
    X_train_clean = pipeline.fit_transform(X_train)
    X_test_clean = pipeline.transform(X_test)   # transform only, never fit

    save_pipeline(pipeline, "artifacts/cleaning_pipeline.joblib")
    # later, once new data arrives:
    pipeline = load_pipeline("artifacts/cleaning_pipeline.joblib")
    X_new_clean = pipeline.transform(X_new)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps each feature at [Q1 - factor*IQR, Q3 + factor*IQR], learned from
    the fit() data only. Values outside that range are clipped, not
    dropped -- this preserves row alignment with labels and avoids
    silently shrinking the dataset, which matters when data is already
    scarce (as it is on this project right now).
    """

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = (q1 - self.iqr_factor * iqr).values
        self.upper_bounds_ = (q3 + self.iqr_factor * iqr).values
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        X_capped = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_capped.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)


def build_cleaning_pipeline(
    imputation_strategy: str = "median",
    iqr_factor: float = 1.5,
    scaler: str = "robust",
) -> Pipeline:
    """
    Build the standardized cleaning pipeline.

    Args:
        imputation_strategy: "median" (default, robust to skew), "mean", or "most_frequent"
        iqr_factor: how wide the outlier-capping range is (1.5 = standard Tukey fence)
        scaler: "robust" (default -- uses median/IQR, resistant to remaining outliers)
                or "standard" (uses mean/std -- only use if you're confident
                outliers are fully handled upstream)
    """
    scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()

    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
        ("outlier_capper", OutlierCapper(iqr_factor=iqr_factor)),
        ("scaler", scaler_obj),
    ])


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Persist a fitted pipeline so it can be reused without re-fitting."""
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    """Load a previously fitted pipeline."""
    return joblib.load(path)


def clean_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
    imputation_strategy: str = "median",
    iqr_factor: float = 1.5,
    scaler: str = "robust",
):
    """
    Convenience wrapper: build, fit on X_train, transform X_train (and
    X_test if given). Returns clean DataFrames with original column
    names and index preserved, plus the fitted pipeline for reuse.
    """
    pipeline = build_cleaning_pipeline(imputation_strategy, iqr_factor, scaler)

    X_train_arr = pipeline.fit_transform(X_train)
    X_train_clean = pd.DataFrame(X_train_arr, columns=X_train.columns, index=X_train.index)

    if X_test is not None:
        X_test_arr = pipeline.transform(X_test)
        X_test_clean = pd.DataFrame(X_test_arr, columns=X_test.columns, index=X_test.index)
        return X_train_clean, X_test_clean, pipeline

    return X_train_clean, pipeline
