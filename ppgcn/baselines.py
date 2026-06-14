from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC

from .evaluation import classification_metrics


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def build_svm(seed: int, kernel: str = "linear", c: float = 1.0, max_iter: int = 10000):
    if kernel == "linear":
        return LinearSVC(C=c, dual=False, max_iter=max_iter, random_state=seed)
    return make_pipeline(SVC(C=c, kernel=kernel, gamma="scale", random_state=seed))


def run_svm_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    kernel: str = "linear",
    c: float = 1.0,
    max_iter: int = 10000,
) -> tuple[dict[str, float], np.ndarray]:
    start = time.perf_counter()
    model = build_svm(seed=seed, kernel=kernel, c=c, max_iter=max_iter)
    model.fit(flatten_windows(x_train), y_train)
    preds = model.predict(flatten_windows(x_test))
    metrics = classification_metrics(y_test, preds)
    metrics.update(
        {
            "kernel": kernel,
            "C": float(c),
            "fit_predict_seconds": float(time.perf_counter() - start),
        }
    )
    return metrics, preds


def run_pca_svm_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    n_components: int = 128,
    c: float = 1.0,
    max_iter: int = 10000,
) -> tuple[dict[str, float], np.ndarray]:
    start = time.perf_counter()
    flat_train = flatten_windows(x_train)
    flat_test = flatten_windows(x_test)
    n_components = max(1, min(n_components, flat_train.shape[0] - 1, flat_train.shape[1]))
    model = make_pipeline(
        PCA(n_components=n_components, random_state=seed),
        LinearSVC(C=c, dual=False, max_iter=max_iter, random_state=seed),
    )
    model.fit(flat_train, y_train)
    preds = model.predict(flat_test)
    metrics = classification_metrics(y_test, preds)
    metrics.update(
        {
            "n_components": int(n_components),
            "C": float(c),
            "fit_predict_seconds": float(time.perf_counter() - start),
        }
    )
    return metrics, preds


def run_random_forest_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    n_estimators: int = 200,
    max_depth: int | None = None,
    n_jobs: int = -1,
) -> tuple[dict[str, float], np.ndarray]:
    start = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(flatten_windows(x_train), y_train)
    preds = model.predict(flatten_windows(x_test))
    metrics = classification_metrics(y_test, preds)
    metrics.update(
        {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "fit_predict_seconds": float(time.perf_counter() - start),
        }
    )
    return metrics, preds
