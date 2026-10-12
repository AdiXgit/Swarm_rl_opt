"""Shared agent interface: prediction type, XGBoost fit/predict/save/load.

Every specialist agent (Direction, Volatility, Regime) is a thin subclass of
`XGBClassifierAgent` that only fixes `AGENT_NAME` and `CLASSES`. The Greeks
agent is deterministic and does not use this base (see `greeks_agent.py`).
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_VERSION = "xgb_v1"

#: Numerical tolerance for "probabilities sum to one" and "each in [0, 1]".
PROB_TOLERANCE = 1e-6


class AgentError(Exception):
    pass


@dataclass(frozen=True)
class AgentPrediction:
    """One agent's output for one observation timestamp.

    `confidence` is the gap between the top-1 and top-2 class probabilities
    (an uncalibrated margin, not a probability of being correct -- nothing
    here has been calibrated against held-out outcomes). `data_quality`
    records whatever made this prediction less than fully trustworthy
    (missing features, insufficient lookback, degraded inputs), never
    silently absorbed into the probabilities themselves.
    """

    agent_name: str
    model_version: str
    observation_timestamp: str
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    data_quality: dict[str, bool | str]

    def validate(self) -> None:
        probs = list(self.probabilities.values())
        if not all(np.isfinite(p) for p in probs):
            raise AgentError(f"{self.agent_name}: non-finite probability {self.probabilities}")
        if any(p < -PROB_TOLERANCE or p > 1 + PROB_TOLERANCE for p in probs):
            raise AgentError(f"{self.agent_name}: probability out of [0,1] {self.probabilities}")
        if abs(sum(probs) - 1.0) > 1e-3:
            raise AgentError(f"{self.agent_name}: probabilities sum to {sum(probs)}, not 1")
        if self.predicted_class not in self.probabilities:
            raise AgentError(f"{self.agent_name}: predicted_class not in probabilities")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentMetadata:
    agent_name: str
    model_version: str
    classes: list[str]
    feature_names: list[str]
    feature_schema_version: str
    target_definition: str
    train_range: list[str]
    seed: int
    xgboost_version: str
    numpy_version: str
    pandas_version: str
    python_version: str
    trained_at: str
    params: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMetadata":
        return cls(**d)


class XGBClassifierAgent:
    """XGBoost multiclass classifier with a validated, versioned interface.

    Serialization uses XGBoost's own JSON model format plus a plain-JSON
    metadata sidecar -- never pickle -- so loading a saved agent never
    executes arbitrary code.
    """

    AGENT_NAME: str = "base"
    CLASSES: tuple[str, ...] = ()
    TARGET_DEFINITION: str = ""

    def __init__(self, seed: int = 0, params: dict | None = None):
        self.seed = seed
        # Deliberately shallow/few trees: TRAIN has ~190 weekly rows (see
        # docs/EXPERIMENTS.md "Sample Size"), so a deep/large model memorises
        # the training set rather than generalising. These are a documented
        # a-priori default for a small weekly-cadence sample, not a value
        # tuned against VALIDATION or TEST accuracy.
        self.params = {
            "n_estimators": 50,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "n_jobs": 1,
            **(params or {}),
        }
        self._model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.feature_schema_version: str = ""
        self.train_range: tuple[str, str] = ("", "")
        self._missing_classes_in_fit: list[str] = []

    @property
    def classes(self) -> tuple[str, ...]:
        return self.CLASSES

    # -- fit ----------------------------------------------------------------- #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_schema_version: str,
        train_range: tuple[str, str],
    ) -> "XGBClassifierAgent":
        unknown = set(y.unique()) - set(self.CLASSES)
        if unknown:
            raise AgentError(f"{self.AGENT_NAME}: labels outside CLASSES: {unknown}")
        present = set(y.unique())
        missing_classes = set(self.CLASSES) - present
        # Not fatal: XGBoost's objective is fit over whatever classes are
        # present. Recorded so a fold lacking a class is visible, not silent.
        self._missing_classes_in_fit = sorted(missing_classes)

        class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}
        y_idx = y.map(class_to_idx)
        if y_idx.isna().any():
            raise AgentError(f"{self.AGENT_NAME}: label mapping produced NaN")

        self.feature_names = list(X.columns)
        self.feature_schema_version = feature_schema_version
        self.train_range = train_range

        # The low-level Booster API is used rather than the XGBClassifier
        # sklearn wrapper: the wrapper infers its class set from the labels
        # actually seen and raises if a class is absent from this TRAIN fold
        # (a real possibility with ~190 weekly rows across 3-5 classes, see
        # docs/EXPERIMENTS.md "Sample Size"). The Booster just needs
        # `num_class`-wide softprob output and integer indices into it, so a
        # missing class degrades to "near-zero probability", not a crash.
        dtrain = xgb.DMatrix(
            X[self.feature_names].to_numpy(dtype=float), label=y_idx.to_numpy()
        )
        booster_params = {
            "objective": "multi:softprob",
            "num_class": len(self.CLASSES),
            "seed": self.seed,
            "max_depth": self.params["max_depth"],
            "eta": self.params["learning_rate"],
            "subsample": self.params["subsample"],
            "colsample_bytree": self.params["colsample_bytree"],
            "lambda": self.params["reg_lambda"],
            "nthread": self.params["n_jobs"],
        }
        self._model = xgb.train(booster_params, dtrain, num_boost_round=self.params["n_estimators"])
        return self

    # -- predict --------------------------------------------------------------#
    def predict(self, X: pd.DataFrame, timestamps: pd.Series) -> list[AgentPrediction]:
        if self._model is None:
            raise AgentError(f"{self.AGENT_NAME}: model not fit or loaded")
        missing_cols = [c for c in self.feature_names if c not in X.columns]
        if missing_cols:
            raise AgentError(f"{self.AGENT_NAME}: missing required features {missing_cols}")

        ordered = X[self.feature_names]
        nan_mask = ordered.isna().any(axis=1).to_numpy()
        filled = ordered.fillna(0.0).to_numpy(dtype=float)
        proba = self._model.predict(xgb.DMatrix(filled)).reshape(len(ordered), len(self.CLASSES))

        out = []
        for i in range(len(ordered)):
            probs = {c: float(proba[i, j]) for j, c in enumerate(self.CLASSES)}
            order = sorted(probs.values(), reverse=True)
            confidence = float(order[0] - order[1]) if len(order) > 1 else float(order[0])
            predicted = max(probs, key=probs.get)
            dq: dict[str, bool | str] = {"had_missing_feature": bool(nan_mask[i])}
            pred = AgentPrediction(
                agent_name=self.AGENT_NAME,
                model_version=MODEL_VERSION,
                observation_timestamp=pd.Timestamp(timestamps.iloc[i]).isoformat(),
                predicted_class=predicted,
                probabilities=probs,
                confidence=confidence,
                data_quality=dq,
            )
            pred.validate()
            out.append(pred)
        return out

    # -- persistence ----------------------------------------------------------#
    def save(self, directory: str | Path) -> None:
        if self._model is None:
            raise AgentError(f"{self.AGENT_NAME}: nothing to save, model not fit")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(d / "model.json"))
        meta = AgentMetadata(
            agent_name=self.AGENT_NAME,
            model_version=MODEL_VERSION,
            classes=list(self.CLASSES),
            feature_names=self.feature_names,
            feature_schema_version=self.feature_schema_version,
            target_definition=self.TARGET_DEFINITION,
            train_range=list(self.train_range),
            seed=self.seed,
            xgboost_version=xgb.__version__,
            numpy_version=np.__version__,
            pandas_version=pd.__version__,
            python_version=platform.python_version(),
            trained_at=datetime.now(timezone.utc).isoformat(),
            params=self.params,
        )
        (d / "metadata.json").write_text(json.dumps(meta.to_dict(), indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "XGBClassifierAgent":
        d = Path(directory)
        meta_path, model_path = d / "metadata.json", d / "model.json"
        if not meta_path.exists() or not model_path.exists():
            raise AgentError(f"{cls.AGENT_NAME}: missing artifact in {d}")
        meta = AgentMetadata.from_dict(json.loads(meta_path.read_text()))
        if tuple(meta.classes) != cls.CLASSES:
            raise AgentError(
                f"{cls.AGENT_NAME}: saved classes {meta.classes} incompatible with {cls.CLASSES}"
            )

        agent = cls(seed=meta.seed, params=meta.params)
        model = xgb.Booster()
        model.load_model(str(model_path))
        agent._model = model
        agent.feature_names = meta.feature_names
        agent.feature_schema_version = meta.feature_schema_version
        agent.train_range = tuple(meta.train_range)
        return agent
