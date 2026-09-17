"""Train and persist the frozen scoring artifacts.

Usage::

    python -m ml.train.train --data data/synthetic --out models

Writes ``models/iforest.joblib``, ``models/autoencoder.pt``,
``models/autoencoder_meta.json``, ``models/scaler.joblib`` and
``models/horizon.json``. If no synthetic corpus is present the trainer falls back
to the committed ``data/sample/sample_links.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.generator.gnpy_like import add_horizon_labels
from ml import features as F
from ml.models.autoencoder import AutoencoderScorer
from ml.models.horizon import (
    HorizonHead,
    P_FAIL_THRESHOLD,
    TARGET_NOMINAL_FPR,
    ScoringPipeline,
    combine_scores,
    false_positive_rate,
)
from ml.models.isolation_forest import IsolationForestScorer

SERVICE = "train"
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "sample_links.csv"


def log(msg: str, **kv) -> None:
    record = {"ts": time.time(), "service": SERVICE, "msg": msg}
    record.update(kv)
    print(json.dumps(record, default=str), flush=True)


def load_corpus(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load the synthetic corpus, or fall back to the committed sample CSV."""
    frames: list[pd.DataFrame] = []
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(path))
        if not frames:
            for path in sorted(data_dir.glob("*.csv")):
                frames.append(pd.read_csv(path))
    if frames:
        return pd.concat(frames, ignore_index=True), str(data_dir)

    if not SAMPLE_CSV.exists():
        raise SystemExit(f"no corpus in {data_dir} and no sample CSV at {SAMPLE_CSV}")
    return pd.read_csv(SAMPLE_CSV), str(SAMPLE_CSV)


def build_windows(df: pd.DataFrame, stride: int):
    if "y_horizon" not in df.columns:
        df = add_horizon_labels(df)
    x_stats, x_seq, meta = F.windows_from_dataframe(df, stride=stride)
    if len(meta) == 0:
        raise SystemExit("no complete 60-sample windows in corpus")
    return x_stats, x_seq, meta


def train(
    data_dir: Path,
    out_dir: Path,
    stride: int | None = None,
    epochs: int = 20,
    verbose: bool = False,
) -> dict:
    df, source = load_corpus(data_dir)
    if stride is None:
        stride = 1 if source.endswith(".csv") else 10
    log("corpus_loaded", source=source, rows=int(len(df)), links=int(df["link_id"].nunique()))

    x_stats, x_seq, meta = build_windows(df, stride=stride)
    log("windows_built", n=int(len(meta)), stride=stride)

    nominal_mask = (meta["label"] == "nominal").to_numpy()
    if nominal_mask.sum() < 50:
        raise SystemExit(
            f"need >=50 nominal windows to fit baseline models, got {int(nominal_mask.sum())}"
        )
    log("nominal_windows", n=int(nominal_mask.sum()))

    iforest = IsolationForestScorer.fit(x_stats[nominal_mask])
    log("iforest_fitted", lo=iforest.lo, hi=iforest.hi)

    autoencoder = AutoencoderScorer.fit(x_seq[nominal_mask], epochs=epochs, verbose=verbose)
    log("autoencoder_fitted", err_p98=autoencoder.err_p98)

    s_if = iforest.score(x_stats)
    s_ae = autoencoder.score(x_seq)
    s = combine_scores(s_if, s_ae)
    y = meta["y_horizon"].to_numpy(dtype=int) if "y_horizon" in meta.columns else np.zeros(len(meta), int)

    horizon = HorizonHead.fit(s, y)
    log("horizon_fitted", k=horizon.k, s0=horizon.s0, positives=int(y.sum()))
    horizon.calibrate_threshold(s[nominal_mask])
    log("horizon_calibrated", k=horizon.k, s0=horizon.s0, target_fpr=TARGET_NOMINAL_FPR)

    p_fail = horizon.predict(s)
    fpr = false_positive_rate(p_fail[nominal_mask], P_FAIL_THRESHOLD)
    pos = y == 1
    recall = float(np.mean(p_fail[pos] >= P_FAIL_THRESHOLD)) if pos.any() else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    if_path = iforest.save(out_dir)
    ae_paths = autoencoder.save(out_dir)
    horizon_path = horizon.save(out_dir)

    report = {
        "source": source,
        "windows": int(len(meta)),
        "nominal_windows": int(nominal_mask.sum()),
        "horizon_positives": int(y.sum()),
        "k": float(horizon.k),
        "s0": float(horizon.s0),
        "nominal_fpr_at_0.7": float(fpr),
        "horizon_recall_at_0.7": recall,
        "mean_p_fail_nominal": float(np.mean(p_fail[nominal_mask])),
        "mean_p_fail_positive": float(np.mean(p_fail[pos])) if pos.any() else float("nan"),
        "artifacts": [str(if_path), *[str(p) for p in ae_paths.values()], str(horizon_path)],
    }
    log("train_done", **report)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")

    if not ScoringPipeline.artifacts_present(out_dir):
        raise SystemExit("artifacts missing after training")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train.train", description=__doc__)
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="models")
    parser.add_argument("--stride", type=int, default=None, help="window stride (default 10 parquet / 1 csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    train(
        Path(args.data),
        Path(args.out),
        stride=args.stride,
        epochs=args.epochs,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
