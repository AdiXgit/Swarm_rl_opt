"""Feature manifest: the deterministic contract between training and inference.

Saved alongside every trained agent so inference can verify, before loading
any model, that the feature names/order/dtypes it is about to build match
what the model was actually trained on.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class FeatureSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureManifest:
    feature_schema_version: str
    feature_names: list[str]
    max_lookback_days: int
    target_schema_version: str
    missing_value_policy: str = (
        "NaN preserved; agents fill NaN with 0.0 at predict time and flag "
        "had_missing_feature=True rather than imputing silently"
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureManifest":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def assert_compatible(self, feature_names: list[str], feature_schema_version: str) -> None:
        if feature_schema_version != self.feature_schema_version:
            raise FeatureSchemaError(
                f"feature_schema_version mismatch: manifest={self.feature_schema_version} "
                f"actual={feature_schema_version}"
            )
        if list(feature_names) != self.feature_names:
            raise FeatureSchemaError(
                f"feature name/order mismatch: manifest={self.feature_names} actual={feature_names}"
            )
