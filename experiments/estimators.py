"""Train-only preprocessing and estimators shared by the seven notebooks."""

import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def structured_preprocessor(scale=False):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True,
                                      keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=5, max_categories=64,
                                 sparse_output=True, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(steps), NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ], sparse_threshold=1.0)


def native_features(frame, categories=None):
    """Map unseen levels to missing using exclusively training category levels."""
    features = frame[NUMERIC + CATEGORICAL].copy()
    for column in CATEGORICAL:
        features[column] = features[column].fillna("__MISSING__").astype(str)
    if categories is None:
        categories = {column: sorted(features[column].unique().tolist()) for column in CATEGORICAL}
    for column in CATEGORICAL:
        features[column] = pd.Categorical(features[column], categories=categories[column])
    return features, categories


def catboost_features(frame):
    features = frame[NUMERIC + CATEGORICAL].copy()
    for column in CATEGORICAL:
        features[column] = features[column].fillna("__MISSING__").astype(str)
    return features


def text_pipeline(alpha, ngram_range, max_features, min_df, max_df):
    from sklearn.linear_model import Ridge
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                  min_df=min_df, max_df=max_df, sublinear_tf=True,
                                  strip_accents="unicode", dtype=np.float64)),
        ("regressor", Ridge(alpha=alpha, solver="lsqr", tol=1e-5)),
    ])


def redact_text(value):
    """Remove explicit price cues from model text without learning a vocabulary."""
    text = unicodedata.normalize("NFKD", str(value).lower().replace("đ", "d"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    money = r"\b(?:gia|ty|ti|trieu|tr|vnd|dong|usd|price|billion|million)\b|[$₫€]"
    clauses = re.split(r"[\n.!?;,]+", text)
    text = " ".join(clause for clause in clauses if not re.search(money, clause))
    text = re.sub(r"\[phone_number\]|\b\w*\d\w*\b", " ", text)
    return " ".join(text.split())


def text_inputs(frame):
    """Use prepared text or build TF-IDF input from the EDA text fields."""
    if "text" in frame:
        return frame.text.fillna("").astype(str)
    columns = [name for name in ["name", "description"] if name in frame]
    if not columns:
        raise ValueError("TF-IDF requires prepared text or name/description columns")
    return frame[columns].fillna("").astype(str).agg(" . ".join, axis=1).map(redact_text)


class MedianPricePerSqm(RegressorMixin, BaseEstimator):
    """Hierarchical training medians, with province-qualified district keys."""

    def fit(self, X, y):
        frame = X.copy()
        price = np.asarray(y, dtype=float)
        area = pd.to_numeric(frame.area, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(price) & (price > 0) & np.isfinite(area) & (area > 0)
        if not valid.any():
            raise ValueError("At least one positive finite training price and area is required")
        frame = self._keys(frame.loc[valid].copy())
        frame["ppm"] = price[valid] / area[valid]
        self.area_median_ = float(np.median(area[valid]))
        self.global_median_ = float(frame.ppm.median())
        self.group_columns_ = [
            ["province_name", "district_name", "property_type_name"],
            ["province_name", "property_type_name"], ["property_type_name"],
        ]
        self.medians_ = [frame.groupby(columns, dropna=False).ppm.median().to_dict()
                         for columns in self.group_columns_]
        return self

    @staticmethod
    def _keys(frame):
        frame = frame.copy()
        for column in ["province_name", "district_name", "property_type_name"]:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        return frame

    def predict_with_fallback(self, X):
        frame = self._keys(X)
        ppm = np.full(len(frame), np.nan)
        level = np.full(len(frame), "global", dtype=object)
        for columns, medians, label in zip(self.group_columns_, self.medians_,
                                          ["district_type", "province_type", "type"]):
            keys = list(frame[columns].itertuples(index=False, name=None))
            if len(columns) == 1:
                keys = [key[0] for key in keys]
            values = np.array([medians.get(key, np.nan) for key in keys])
            use = np.isnan(ppm) & np.isfinite(values)
            ppm[use], level[use] = values[use], label
        ppm = np.where(np.isfinite(ppm), ppm, self.global_median_)
        area = pd.to_numeric(frame.area, errors="coerce").to_numpy(dtype=float)
        area = np.where(np.isfinite(area) & (area > 0), area, self.area_median_)
        return ppm * area, level

    def predict(self, X):
        return self.predict_with_fallback(X)[0]
