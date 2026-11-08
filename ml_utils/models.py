"""Model factories. Skeleton: one function per model, each returning an unfitted estimator/pipeline.

Imports of heavy libraries are done inside the functions so that importing this module is cheap.
Naming: build_<algorithm>() ; artifacts are saved as models/<algorithm>_<target>_v<N>.<ext>.
"""

from .config import RANDOM_STATE


def build_dummy_median():
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="median")


def build_ridge(alpha: float = 1.0):
    raise NotImplementedError("Add preprocessing (encoding + scaling) in notebook 04.")


def build_lightgbm(**params):
    raise NotImplementedError("Notebook 05.")


def build_catboost(**params):
    raise NotImplementedError("Notebook 05.")


def build_xgboost(**params):
    raise NotImplementedError("Notebook 05.")


__all__ = ["RANDOM_STATE", "build_dummy_median", "build_ridge", "build_lightgbm", "build_catboost", "build_xgboost"]
