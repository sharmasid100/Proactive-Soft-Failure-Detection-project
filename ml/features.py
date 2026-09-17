"""Feature engineering: 60x5 sequence tensor plus the 8 Isolation Forest stats.

The feature *frame* is the JSONL contract between the C++ ingest and the Python
infer service (see CLAUDE.md 4.2). This module is the Python-side reference
implementation, used for training and for scoring frames received over TCP.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

WINDOW_N = 60
SAMPLE_HZ = 1

BER_FLOOR = 1e-15

# Autoencoder channels, fixed order.
SEQ_CHANNELS: tuple[str, ...] = (
    "osnr_db",
    "log10_ber",
    "laser_bias_ma",
    "edfa_pump_ma",
    "rx_power_dbm",
)

# Isolation Forest features, fixed order.
FEATURE_NAMES: tuple[str, ...] = (
    "osnr_mean",
    "osnr_std",
    "osnr_slope",
    "log10_ber_mean",
    "log10_ber_std",
    "log10_ber_slope",
    "laser_bias_mean",
    "laser_bias_slope",
)

EDFA_PUMP_DEFAULT = 0.0
RX_POWER_DEFAULT = -99.0


def log10_ber(ber) -> np.ndarray:
    """``log10(max(ber, 1e-15))`` (CLAUDE.md 22)."""
    arr = np.asarray(ber, dtype=float)
    return np.log10(np.maximum(arr, BER_FLOOR))


def ols_slope(values: Sequence[float]) -> float:
    """Ordinary-least-squares slope of ``values`` against index ``0..n-1``."""
    y = np.asarray(values, dtype=float)
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0.0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)


def _slope_axis(mat: np.ndarray) -> np.ndarray:
    """Vectorised OLS slope along axis 1 of a ``(N, n, ...)`` array."""
    n = mat.shape[1]
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    shape = [1] * mat.ndim
    shape[1] = n
    xc_b = xc.reshape(shape)
    yc = mat - mat.mean(axis=1, keepdims=True)
    return (xc_b * yc).sum(axis=1) / denom


def sequences_from_samples(samples: Sequence[dict]) -> dict[str, list[float]]:
    """Build the ``seq`` block of a feature frame from raw samples."""
    osnr = [float(s["osnr_db"]) for s in samples]
    ber = [float(s["ber"]) for s in samples]
    bias = [float(s["laser_bias_ma"]) for s in samples]
    pump = [
        EDFA_PUMP_DEFAULT if s.get("edfa_pump_ma") is None else float(s["edfa_pump_ma"])
        for s in samples
    ]
    rx = [
        RX_POWER_DEFAULT if s.get("rx_power_dbm") is None else float(s["rx_power_dbm"])
        for s in samples
    ]
    return {
        "osnr_db": osnr,
        "log10_ber": [float(v) for v in log10_ber(ber)],
        "laser_bias_ma": bias,
        "edfa_pump_ma": pump,
        "rx_power_dbm": rx,
    }


def compute_stats(seq: dict[str, Sequence[float]]) -> dict[str, float]:
    """The 8 stats of a feature frame."""
    osnr = np.asarray(seq["osnr_db"], dtype=float)
    lber = np.asarray(seq["log10_ber"], dtype=float)
    bias = np.asarray(seq["laser_bias_ma"], dtype=float)
    return {
        "osnr_mean": float(osnr.mean()),
        "osnr_std": float(osnr.std()),
        "osnr_slope": ols_slope(osnr),
        "log10_ber_mean": float(lber.mean()),
        "log10_ber_std": float(lber.std()),
        "log10_ber_slope": ols_slope(lber),
        "laser_bias_mean": float(bias.mean()),
        "laser_bias_slope": ols_slope(bias),
    }


def make_frame(samples: Sequence[dict], window_n: int = WINDOW_N) -> dict:
    """Build a complete feature frame from exactly ``window_n`` raw samples."""
    if len(samples) != window_n:
        raise ValueError(f"expected {window_n} samples, got {len(samples)}")
    last = samples[-1]
    seq = sequences_from_samples(samples)
    return {
        "ts_unix_ms": int(last["ts_unix_ms"]),
        "link_id": str(last["link_id"]),
        "channel_id": str(last.get("channel_id", "C1")),
        "n": window_n,
        "seq": seq,
        "stats": compute_stats(seq),
    }


def stats_vector(stats: dict[str, float]) -> np.ndarray:
    """The 8-vector in :data:`FEATURE_NAMES` order."""
    return np.asarray([float(stats[name]) for name in FEATURE_NAMES], dtype=float)


def frame_stats_vector(frame: dict) -> np.ndarray:
    stats = frame.get("stats") or compute_stats(frame["seq"])
    return stats_vector(stats)


def frame_matrix(frame: dict) -> np.ndarray:
    """The ``[60, 5]`` autoencoder input matrix (unscaled)."""
    seq = frame["seq"]
    return np.stack([np.asarray(seq[ch], dtype=float) for ch in SEQ_CHANNELS], axis=1)


def windows_from_dataframe(
    df: pd.DataFrame,
    window_n: int = WINDOW_N,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Slice a raw-sample DataFrame into windows.

    Returns ``(X_stats, X_seq, meta)`` where ``X_stats`` is ``(N, 8)``, ``X_seq``
    is ``(N, window_n, 5)`` and ``meta`` carries ``link_id``, ``channel_id``,
    ``ts_unix_ms`` (last sample of the window), plus ``label`` / ``y_horizon``
    when present in ``df``. Incomplete windows are dropped.
    """
    if "channel_id" not in df.columns:
        df = df.assign(channel_id="C1")
    work = df.copy()
    work["log10_ber"] = log10_ber(work["ber"].to_numpy())
    if "edfa_pump_ma" not in work.columns:
        work["edfa_pump_ma"] = EDFA_PUMP_DEFAULT
    if "rx_power_dbm" not in work.columns:
        work["rx_power_dbm"] = RX_POWER_DEFAULT
    work["edfa_pump_ma"] = work["edfa_pump_ma"].fillna(EDFA_PUMP_DEFAULT)
    work["rx_power_dbm"] = work["rx_power_dbm"].fillna(RX_POWER_DEFAULT)

    seq_blocks: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []

    for (link_id, channel_id), group in work.groupby(["link_id", "channel_id"], sort=True):
        group = group.sort_values("ts_unix_ms")
        if len(group) < window_n:
            continue
        mat = group[list(SEQ_CHANNELS)].to_numpy(dtype=float)
        views = np.lib.stride_tricks.sliding_window_view(mat, window_n, axis=0)
        # views: (n_windows, 5, window_n) -> (n_windows, window_n, 5)
        views = np.transpose(views, (0, 2, 1))[::stride]
        seq_blocks.append(np.ascontiguousarray(views))

        tail = group.iloc[window_n - 1 :: 1]
        tail = tail.iloc[::stride]
        cols = {
            "link_id": tail["link_id"].to_numpy(),
            "channel_id": tail["channel_id"].to_numpy(),
            "ts_unix_ms": tail["ts_unix_ms"].to_numpy(),
        }
        for extra in ("label", "y_horizon", "hard_fail"):
            if extra in tail.columns:
                cols[extra] = tail[extra].to_numpy()
        meta_rows.append(pd.DataFrame(cols))

    if not seq_blocks:
        empty_meta = pd.DataFrame(columns=["link_id", "channel_id", "ts_unix_ms"])
        return (
            np.empty((0, len(FEATURE_NAMES))),
            np.empty((0, window_n, len(SEQ_CHANNELS))),
            empty_meta,
        )

    x_seq = np.concatenate(seq_blocks, axis=0)
    meta = pd.concat(meta_rows, ignore_index=True)
    return stats_from_sequences(x_seq), x_seq, meta


def stats_from_sequences(x_seq: np.ndarray) -> np.ndarray:
    """Vectorised 8-feature stats for a ``(N, window_n, 5)`` batch."""
    if x_seq.size == 0:
        return np.empty((0, len(FEATURE_NAMES)))
    osnr = x_seq[:, :, 0]
    lber = x_seq[:, :, 1]
    bias = x_seq[:, :, 2]
    slopes = _slope_axis(x_seq)  # (N, 5)
    return np.stack(
        [
            osnr.mean(axis=1),
            osnr.std(axis=1),
            slopes[:, 0],
            lber.mean(axis=1),
            lber.std(axis=1),
            slopes[:, 1],
            bias.mean(axis=1),
            slopes[:, 2],
        ],
        axis=1,
    )


def iter_frames(
    samples: Iterable[dict], window_n: int = WINDOW_N
) -> Iterator[dict]:
    """Stream raw samples of one key into feature frames (one per new sample)."""
    buffer: list[dict] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) > window_n:
            buffer.pop(0)
        if len(buffer) == window_n:
            yield make_frame(buffer, window_n=window_n)
