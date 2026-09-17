"""Isolation Forest anomaly scorer over the 8 window statistics.

Fitted on **nominal** windows only; the raw score is min-max calibrated against a
nominal holdout so that ``s_if`` lands in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

N_ESTIMATORS = 200
CONTAMINATION = 0.02
RANDOM_STATE = 42
CALIB_PERCENTILE = 99.5

ARTIFACT_NAME = "iforest.joblib"


@dataclass
class IsolationForestScorer:
    """Isolation Forest + [0, 1] calibration."""

    model: IsolationForest
    lo: float
    hi: float

    # --- scoring ---------------------------------------------------------- #
    def raw_scores(self, x: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return -self.model.decision_function(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        span = self.hi - self.lo
        if span <= 0:
            span = 1e-9
        return np.clip((self.raw_scores(x) - self.lo) / span, 0.0, 1.0)

    def score_one(self, x: np.ndarray) -> float:
        return float(self.score(np.asarray(x, dtype=float).reshape(1, -1))[0])

    # --- construction ----------------------------------------------------- #
    @classmethod
    def fit(
        cls,
        x_nominal: np.ndarray,
        holdout_frac: float = 0.2,
        calib_percentile: float = CALIB_PERCENTILE,
        random_state: int = RANDOM_STATE,
    ) -> "IsolationForestScorer":
        x_nominal = np.asarray(x_nominal, dtype=float)
        if x_nominal.ndim != 2:
            raise ValueError("x_nominal must be 2-D (n_windows, 8)")
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(x_nominal))
        n_hold = max(1, int(round(holdout_frac * len(x_nominal))))
        hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]
        if len(fit_idx) == 0:  # tiny dataset: fit on everything
            fit_idx, hold_idx = idx, idx

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=random_state,
        )
        model.fit(x_nominal[fit_idx])

        holdout_raw = -model.decision_function(x_nominal[hold_idx])
        lo = float(np.median(holdout_raw))
        hi = float(np.percentile(holdout_raw, calib_percentile))
        if hi <= lo:
            hi = lo + 1e-6
        return cls(model=model, lo=lo, hi=hi)

    # --- persistence ------------------------------------------------------ #
    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / ARTIFACT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "lo": self.lo, "hi": self.hi}, path, compress=3)
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> "IsolationForestScorer":
        path = Path(out_dir)
        if path.is_dir():
            path = path / ARTIFACT_NAME
        blob = joblib.load(path)
        return cls(model=blob["model"], lo=float(blob["lo"]), hi=float(blob["hi"]))
