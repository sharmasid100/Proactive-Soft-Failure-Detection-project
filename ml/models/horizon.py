"""Horizon head: combine anomaly scores into ``P(hard failure within 600 s)``.

::

    s = 0.5 * s_if + 0.5 * s_ae
    p_fail_10m = 1 / (1 + exp(-k * (s - s0)))

``k`` and ``s0`` are fitted with logistic regression on synthetic labelled
windows (label = hard failure within the next 600 s). This is the only supervised
piece and it only ever sees synthetic labels; the live path uses frozen weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml import features as F

HORIZON_S = 600
DEFAULT_K = 12.0
DEFAULT_S0 = 0.55
P_FAIL_THRESHOLD = 0.7
# Calibration gate is FPR <= 0.02; aim well below it so unseen nominal traffic
# still clears the gate.
TARGET_NOMINAL_FPR = 0.002

ARTIFACT_NAME = "horizon.json"

IF_WEIGHT = 0.5
AE_WEIGHT = 0.5


def combine_scores(s_if, s_ae):
    """``s = 0.5 * s_if + 0.5 * s_ae``."""
    return IF_WEIGHT * np.asarray(s_if, dtype=float) + AE_WEIGHT * np.asarray(s_ae, dtype=float)


@dataclass
class HorizonHead:
    """Logistic map from the combined anomaly score to ``p_fail_10m``."""

    k: float = DEFAULT_K
    s0: float = DEFAULT_S0
    horizon_s: int = HORIZON_S

    def predict(self, s):
        s_arr = np.asarray(s, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.k * (s_arr - self.s0)))

    def predict_one(self, s: float) -> float:
        return float(self.predict(np.asarray([s], dtype=float))[0])

    @classmethod
    def fit(cls, s, y, horizon_s: int = HORIZON_S) -> "HorizonHead":
        """Fit ``k``/``s0`` with logistic regression on labelled windows."""
        from sklearn.linear_model import LogisticRegression

        s_arr = np.asarray(s, dtype=float).reshape(-1, 1)
        y_arr = np.asarray(y, dtype=int).ravel()
        if len(np.unique(y_arr)) < 2:
            return cls(horizon_s=horizon_s)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(s_arr, y_arr)
        k = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        if abs(k) < 1e-9:
            return cls(horizon_s=horizon_s)
        return cls(k=k, s0=-intercept / k, horizon_s=horizon_s)

    def score_at_probability(self, p: float) -> float:
        """The combined score ``s`` at which ``predict(s) == p``."""
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return self.s0 + float(np.log(p / (1.0 - p))) / self.k

    def calibrate_threshold(
        self,
        nominal_s,
        target_fpr: float = TARGET_NOMINAL_FPR,
        threshold: float = P_FAIL_THRESHOLD,
    ) -> "HorizonHead":
        """Shift ``s0`` right until the nominal FPR at ``threshold`` meets the gate.

        The logistic fit maximises likelihood, which does not by itself guarantee
        the calibration gate (FPR <= 0.02 at ``p_fail_10m >= 0.7``). This nudges
        the operating point so the gate also holds on windows the fit never saw.
        """
        s_arr = np.asarray(nominal_s, dtype=float)
        if s_arr.size == 0 or self.k <= 0:
            return self
        required = float(np.quantile(s_arr, 1.0 - target_fpr))
        offset = float(np.log(threshold / (1.0 - threshold))) / self.k
        s0_needed = required - offset
        if s0_needed > self.s0:
            self.s0 = s0_needed
        return self

    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir)
        if path.suffix != ".json":
            path = path / ARTIFACT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "k": float(self.k),
                    "s0": float(self.s0),
                    "horizon_s": int(self.horizon_s),
                    "if_weight": IF_WEIGHT,
                    "ae_weight": AE_WEIGHT,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> "HorizonHead":
        path = Path(out_dir)
        if path.is_dir():
            path = path / ARTIFACT_NAME
        blob = json.loads(path.read_text())
        return cls(
            k=float(blob["k"]),
            s0=float(blob["s0"]),
            horizon_s=int(blob.get("horizon_s", HORIZON_S)),
        )


@dataclass
class ScoringPipeline:
    """Frozen Isolation Forest + autoencoder + horizon head."""

    iforest: object
    autoencoder: object
    horizon: HorizonHead

    @classmethod
    def load(cls, models_dir: str | Path) -> "ScoringPipeline":
        from ml.models.autoencoder import AutoencoderScorer
        from ml.models.isolation_forest import IsolationForestScorer

        models_dir = Path(models_dir)
        return cls(
            iforest=IsolationForestScorer.load(models_dir),
            autoencoder=AutoencoderScorer.load(models_dir),
            horizon=HorizonHead.load(models_dir),
        )

    @staticmethod
    def required_artifacts(models_dir: str | Path) -> list[Path]:
        from ml.models.autoencoder import META_NAME, SCALER_NAME, WEIGHTS_NAME
        from ml.models.isolation_forest import ARTIFACT_NAME as IF_NAME

        base = Path(models_dir)
        return [base / n for n in (IF_NAME, WEIGHTS_NAME, META_NAME, SCALER_NAME, ARTIFACT_NAME)]

    @classmethod
    def artifacts_present(cls, models_dir: str | Path) -> bool:
        return all(p.exists() for p in cls.required_artifacts(models_dir))

    def score_batch(self, x_stats: np.ndarray, x_seq: np.ndarray) -> dict[str, np.ndarray]:
        s_if = np.asarray(self.iforest.score(x_stats), dtype=float)
        s_ae = np.asarray(self.autoencoder.score(x_seq), dtype=float)
        s = combine_scores(s_if, s_ae)
        return {
            "s_if": s_if,
            "s_ae": s_ae,
            "s": s,
            "p_fail_10m": np.asarray(self.horizon.predict(s), dtype=float),
        }

    def score_frame(self, frame: dict) -> dict[str, float]:
        """Score a single feature frame (the JSONL contract from ingest)."""
        x_stats = F.frame_stats_vector(frame).reshape(1, -1)
        x_seq = F.frame_matrix(frame)[None, :, :]
        out = self.score_batch(x_stats, x_seq)
        return {
            "s_if": float(out["s_if"][0]),
            "s_ae": float(out["s_ae"][0]),
            "p_fail_10m": float(out["p_fail_10m"][0]),
        }


def false_positive_rate(p_fail: np.ndarray, threshold: float = P_FAIL_THRESHOLD) -> float:
    """Fraction of nominal windows at or above the operating threshold."""
    arr = np.asarray(p_fail, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= threshold))
