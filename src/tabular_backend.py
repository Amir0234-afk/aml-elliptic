# src/tabular_backend.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Union, Literal

import numpy as np

MaxFeat = Union[float, Literal["sqrt", "log2"]]


def _compute_sample_weight(y: np.ndarray, class_weight: Optional[dict[int, float]]) -> Optional[np.ndarray]:
    if not class_weight:
        return None
    y = np.asarray(y, dtype=np.int64)
    w = np.ones_like(y, dtype=np.float32)
    for cls, cw in class_weight.items():
        w[y == int(cls)] = float(cw)
    return w


@dataclass
class TabularBackend:
    name: str

    def train_lr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        C: float,
        max_iter: int,
        tol: float,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def train_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_estimators: int,
        max_features: MaxFeat,
        max_depth: int | None,
        min_samples_leaf: int,
        min_samples_split: int = 2,
        max_samples: float | None = None,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def predict_proba_positive(self, model: Any, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_label(self, model: Any, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba_positive(model, X)
        return (p >= float(threshold)).astype(np.int64)


class SklearnBackend(TabularBackend):
    def __init__(self) -> None:
        super().__init__(name="sklearn")

    def train_lr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        C: float,
        max_iter: int,
        tol: float,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(C),
            solver="saga",
            penalty="l2",
            max_iter=int(max_iter),
            tol=float(tol),
            class_weight=class_weight,
            random_state=int(seed),
        )
        model.fit(X, y)
        return model

    def train_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_estimators: int,
        max_features: MaxFeat,
        max_depth: int | None,
        min_samples_leaf: int,
        min_samples_split: int = 2,
        max_samples: float | None = None,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=int(min_samples_split),
            max_samples=max_samples,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=int(seed),
        )
        model.fit(X, y)
        return model

    def predict_proba_positive(self, model: Any, X: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)


class CuMLBackend(TabularBackend):
    def __init__(self) -> None:
        super().__init__(name="cuml")

    def train_lr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        C: float,
        max_iter: int,
        tol: float,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        from cuml.linear_model import LogisticRegression  # type: ignore
        import cupy as cp  # type: ignore

        Xg = cp.asarray(X, dtype=cp.float32)
        yg = cp.asarray(y, dtype=cp.int32)

        sw = _compute_sample_weight(y, class_weight)
        swg = None if sw is None else cp.asarray(sw, dtype=cp.float32)

        model = LogisticRegression(
            C=float(C),
            penalty="l2",
            max_iter=int(max_iter),
            tol=float(tol),
            fit_intercept=True,
            verbose=0,
        )

        try:
            if swg is not None:
                model.fit(Xg, yg, sample_weight=swg)
            else:
                model.fit(Xg, yg)
        except TypeError:
            model.fit(Xg, yg)

        return model

    def train_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_estimators: int,
        max_features: MaxFeat,
        max_depth: int | None,
        min_samples_leaf: int,
        min_samples_split: int = 2,
        max_samples: float | None = None,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        """
        cuML RF caveats (matches your observed errors):
          - some versions reject sample_weight
          - some versions crash on max_depth=None
        This keeps it minimal and stable.
        """
        from cuml.ensemble import RandomForestClassifier  # type: ignore
        import cupy as cp  # type: ignore

        Xg = cp.asarray(X, dtype=cp.float32)
        yg = cp.asarray(y, dtype=cp.int32)

        depth = 16 if max_depth is None else int(max_depth)

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(depth),
            max_features=max_features,  # may be float or "sqrt"/"log2" depending on version
            random_state=int(seed),
        )

        # Do NOT pass sample_weight (your version rejected it).
        # Also, class_weight dict is not consistently supported in cuML RF.
        model.fit(Xg, yg)
        return model

    def predict_proba_positive(self, model: Any, X: np.ndarray) -> np.ndarray:
        import cupy as cp  # type: ignore

        Xg = cp.asarray(X, dtype=cp.float32)
        proba = model.predict_proba(Xg)
        return cp.asnumpy(proba[:, 1]).astype(np.float64)


class XGBoostBackend(TabularBackend):
    def __init__(self) -> None:
        super().__init__(name="xgboost")

    def train_lr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        C: float,
        max_iter: int,
        tol: float,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        import xgboost as xgb  # type: ignore

        sw = _compute_sample_weight(y, class_weight)

        # Rough mapping: larger C => smaller L2
        reg_lambda = float(1.0 / max(float(C), 1e-12))

        model = xgb.XGBClassifier(
            booster="gblinear",
            n_estimators=1,
            reg_lambda=reg_lambda,
            random_state=int(seed),
            device="cuda",
            eval_metric="logloss",
        )
        model.fit(X, y, sample_weight=sw)
        return model

    def train_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_estimators: int,
        max_features: MaxFeat,
        max_depth: int | None,
        min_samples_leaf: int,
        min_samples_split: int = 2,
        max_samples: float | None = None,
        class_weight: Optional[dict[int, float]],
        seed: int,
        **kwargs: Any,
    ) -> Any:
        import xgboost as xgb  # type: ignore

        sw = _compute_sample_weight(y, class_weight)

        depth = 6 if max_depth is None else int(max_depth)
        subsample = 1.0 if max_samples is None else float(max_samples)

        colsample = 1.0
        if isinstance(max_features, float):
            colsample = float(max_features)
        elif isinstance(max_features, str):
            # approximate sqrt/log2; keep bounded
            p = max(X.shape[1], 1)
            if max_features == "sqrt":
                colsample = float(np.sqrt(p) / p)
            elif max_features == "log2":
                colsample = float(np.log2(p) / p)
            colsample = float(np.clip(colsample, 0.05, 1.0))

        model = xgb.XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample),
            learning_rate=0.1,
            reg_lambda=1.0,
            random_state=int(seed),
            device="cuda",
            tree_method="hist",
            eval_metric="logloss",
        )
        model.fit(X, y, sample_weight=sw)
        return model

    def predict_proba_positive(self, model: Any, X: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)


def get_tabular_backend() -> TabularBackend:
    b = os.getenv("AML_TABULAR_BACKEND", "").strip().lower()

    # "auto" => prefer GPU backends if available
    if b in {"", "auto"}:
        try:
            import cuml  # noqa: F401
            import cupy  # noqa: F401
            return CuMLBackend()
        except Exception:
            try:
                import xgboost  # noqa: F401
                return XGBoostBackend()
            except Exception:
                return SklearnBackend()

    if b in {"xgb", "xgboost"}:
        try:
            import xgboost  # noqa: F401
            return XGBoostBackend()
        except Exception:
            return SklearnBackend()

    if b in {"cuml", "rapids", "gpu"}:
        try:
            import cuml  # noqa: F401
            import cupy  # noqa: F401
            return CuMLBackend()
        except Exception:
            return SklearnBackend()

    if b in {"sklearn", "cpu"}:
        return SklearnBackend()

    # unknown value => hard fallback
    return SklearnBackend()
