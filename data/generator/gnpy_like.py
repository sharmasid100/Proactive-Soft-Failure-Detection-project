"""GNPy-like closed-form optical telemetry generator.

This module does **not** call GNPy. It is a closed-form optical toy inspired by
GNPy / Gaussian-noise-model intuition: a nominal operating point plus additive
white noise, with soft-failure impairments superimposed as exponential ramps.

Signals per sample (1 Hz):

* ``osnr_db``        optical signal-to-noise ratio
* ``ber``            pre-FEC bit error rate, mapped monotonically from OSNR
* ``laser_bias_ma``  transceiver laser bias current
* ``edfa_pump_ma``   EDFA pump current
* ``rx_power_dbm``   receiver power

Hard failure (LOS / uncorrectable FEC) is ``osnr_db < 8.0`` OR ``ber > 1e-3``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_START_TS_MS = 1_710_000_000_000

BER_FLOOR = 1e-15
BER_CEIL = 0.5

HARD_OSNR_DB = 8.0
HARD_BER = 1.0e-3
HORIZON_S = 600

LABELS = (
    "nominal",
    "edfa_aging",
    "fiber_pinch",
    "dirty_connector",
    "laser_drift",
    "los",
)

RAW_COLUMNS = (
    "ts_unix_ms",
    "link_id",
    "channel_id",
    "osnr_db",
    "ber",
    "laser_bias_ma",
    "edfa_pump_ma",
    "rx_power_dbm",
    "label",
)

# Impairment -> OSNR coupling coefficients (see CLAUDE.md 5.2).
OSNR_DB_PER_PUMP_MA = 0.04
OSNR_DB_PER_BIAS_MA = 0.03
PINCH_OSNR_PER_LOSS_DB = 0.8
DIRTY_OSNR_PER_LOSS_DB = 0.5
DIRTY_BER_DECADES = 0.5
DIRTY_RIPPLE_DB = 0.2
DIRTY_RIPPLE_HZ = 0.05

LOS_OSNR_DB = 0.0
LOS_BER = 0.1
LOS_RX_DBM = -40.0

_ERFC = np.vectorize(math.erfc, otypes=[float])


@dataclass(frozen=True)
class NominalConstants:
    """Per-link nominal operating point and AWGN sigmas."""

    osnr0_db: float = 18.5
    ber0: float = 1e-6
    laser_bias0_ma: float = 40.0
    edfa_pump0_ma: float = 180.0
    rx0_dbm: float = -12.0
    osnr_sigma_db: float = 0.05
    bias_sigma_ma: float = 0.15
    pump_sigma_ma: float = 0.4
    rx_sigma_db: float = 0.08


@dataclass(frozen=True)
class LinkSpec:
    """One synthetic link trace."""

    link_id: str
    label: str = "nominal"
    channel_id: str = "C1"
    duration_s: int = 1200
    onset_s: float = 0.0
    tau_s: float = 60.0
    severity: float = 0.0
    seed: int = 0
    constants: NominalConstants = field(default_factory=NominalConstants)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}; expected one of {LABELS}")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.tau_s <= 0:
            raise ValueError("tau_s must be positive")


# Severities tuned so that hard failure lands 12-25 min after onset for the
# training corpus (see tests/test_generator.py).
CLASS_SEVERITY = {
    "edfa_aging": 150.0,
    "fiber_pinch": 7.5,
    "dirty_connector": 9.5,
    "laser_drift": 200.0,
    "los": 0.0,
    "nominal": 0.0,
}


# --------------------------------------------------------------------------- #
# Physics helpers
# --------------------------------------------------------------------------- #


def ber_from_osnr(osnr_db):
    """Monotone decreasing OSNR -> pre-FEC BER map (erfc-like).

    ``ber = 0.5 * erfc(sqrt(10**(osnr_db/10) / 2) * 0.5)`` clipped to
    ``[1e-15, 0.5]``. Tests lock this formula.
    """
    osnr = np.asarray(osnr_db, dtype=float)
    lin = np.power(10.0, np.clip(osnr, -100.0, 100.0) / 10.0)
    arg = np.sqrt(lin / 2.0) * 0.5
    ber = 0.5 * _ERFC(arg)
    return np.clip(ber, BER_FLOOR, BER_CEIL)


def ramp_progress(t_s, onset_s: float, tau_s: float):
    """``1 - exp(-(t - onset)/tau)`` for ``t > onset``, else 0."""
    t = np.asarray(t_s, dtype=float)
    delta = np.clip(t - onset_s, 0.0, None)
    prog = 1.0 - np.exp(-delta / float(tau_s))
    return np.where(t > onset_s, prog, 0.0)


def is_hard_fail(osnr_db, ber):
    """Hard failure mask: OSNR below 8 dB or BER above 1e-3."""
    osnr = np.asarray(osnr_db, dtype=float)
    b = np.asarray(ber, dtype=float)
    return (osnr < HARD_OSNR_DB) | (b > HARD_BER)


# --------------------------------------------------------------------------- #
# Trace generation
# --------------------------------------------------------------------------- #


def generate_link(spec: LinkSpec, start_ts_ms: int = DEFAULT_START_TS_MS) -> pd.DataFrame:
    """Generate a single link trace as a DataFrame with :data:`RAW_COLUMNS`."""
    c = spec.constants
    n = int(spec.duration_s)
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(spec.seed)

    osnr = c.osnr0_db + rng.normal(0.0, c.osnr_sigma_db, n)
    bias = c.laser_bias0_ma + rng.normal(0.0, c.bias_sigma_ma, n)
    pump = c.edfa_pump0_ma + rng.normal(0.0, c.pump_sigma_ma, n)
    rx = c.rx0_dbm + rng.normal(0.0, c.rx_sigma_db, n)
    ber_mult = np.ones(n)

    prog = ramp_progress(t, spec.onset_s, spec.tau_s)
    label = spec.label

    if label == "edfa_aging":
        pump_rise = spec.severity * prog
        pump = pump + pump_rise
        osnr = osnr - OSNR_DB_PER_PUMP_MA * pump_rise
    elif label == "fiber_pinch":
        extra_loss = spec.severity * prog
        rx = rx - extra_loss
        osnr = osnr - PINCH_OSNR_PER_LOSS_DB * extra_loss
    elif label == "dirty_connector":
        extra_loss = spec.severity * prog
        ripple = DIRTY_RIPPLE_DB * np.sin(2.0 * np.pi * DIRTY_RIPPLE_HZ * t)
        rx = rx - extra_loss + ripple
        osnr = osnr - DIRTY_OSNR_PER_LOSS_DB * extra_loss + ripple
        ber_mult = np.power(10.0, DIRTY_BER_DECADES * prog)
    elif label == "laser_drift":
        bias_rise = spec.severity * prog
        bias = bias + bias_rise
        osnr = osnr - OSNR_DB_PER_BIAS_MA * bias_rise

    ber = np.clip(ber_from_osnr(osnr) * ber_mult, BER_FLOOR, BER_CEIL)

    if label == "los":
        collapsed = t >= spec.onset_s
        osnr = np.where(collapsed, LOS_OSNR_DB, osnr)
        ber = np.where(collapsed, LOS_BER, ber)
        rx = np.where(collapsed, LOS_RX_DBM, rx)

    ts = start_ts_ms + (t * 1000.0).astype(np.int64)
    return pd.DataFrame(
        {
            "ts_unix_ms": ts,
            "link_id": spec.link_id,
            "channel_id": spec.channel_id,
            "osnr_db": np.round(osnr, 4),
            "ber": ber,
            "laser_bias_ma": np.round(bias, 4),
            "edfa_pump_ma": np.round(pump, 4),
            "rx_power_dbm": np.round(rx, 4),
            "label": label,
        },
        columns=list(RAW_COLUMNS),
    )


def generate_scenario(
    specs: Iterable[LinkSpec], start_ts_ms: int = DEFAULT_START_TS_MS
) -> pd.DataFrame:
    """Generate and concatenate several link traces (sorted by timestamp)."""
    frames = [generate_link(spec, start_ts_ms=start_ts_ms) for spec in specs]
    if not frames:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ts_unix_ms", "link_id"], kind="stable").reset_index(drop=True)


def add_horizon_labels(
    df: pd.DataFrame,
    horizon_s: int = HORIZON_S,
    hard_osnr_db: float = HARD_OSNR_DB,
    hard_ber: float = HARD_BER,
) -> pd.DataFrame:
    """Add ``hard_fail`` and ``y_horizon`` columns.

    ``y_horizon`` is 1 when *any* sample in the next ``horizon_s`` seconds on the
    same ``(link_id, channel_id)`` is a hard failure.
    """
    out = df.copy()
    out["hard_fail"] = (out["osnr_db"] < hard_osnr_db) | (out["ber"] > hard_ber)
    horizon_n = int(horizon_s)

    def _future(group: pd.DataFrame) -> pd.Series:
        group = group.sort_values("ts_unix_ms")
        hf = group["hard_fail"].to_numpy(dtype=float)
        nxt = np.concatenate([hf[1:], [0.0]])
        fut = (
            pd.Series(nxt[::-1])
            .rolling(window=horizon_n, min_periods=1)
            .max()
            .to_numpy()[::-1]
        )
        return pd.Series(fut.astype(int), index=group.index)

    parts = [
        _future(group)
        for _, group in out.groupby(["link_id", "channel_id"], sort=False)
    ]
    out["y_horizon"] = pd.concat(parts).reindex(out.index).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def demo_specs(duration_s: int = 240) -> list[LinkSpec]:
    """Compressed demo schedule (``OPTICS_DEMO=1``).

    ``L1`` is the working path degrading with EDFA aging; hard failure lands
    ~81 s in, i.e. after the first 60 s feature window exists but late enough to
    leave a >=15 s soft-failure window. ``L2`` is protection, ``L3`` a second
    working path that stays nominal.
    """
    return [
        LinkSpec(
            link_id="L1",
            label="edfa_aging",
            duration_s=duration_s,
            onset_s=20.0,
            tau_s=45.0,
            severity=90.0,
            seed=101,
        ),
        LinkSpec(link_id="L2", label="nominal", duration_s=duration_s, seed=102),
        LinkSpec(link_id="L3", label="nominal", duration_s=duration_s, seed=103),
    ]


def training_specs(hours: float = 4.0) -> list[LinkSpec]:
    """Training corpus schedule (see CLAUDE.md 5.3)."""
    nominal_s = int(round(hours * 3600))
    soft_s = 2 * 3600
    los_s = 2 * 3600
    taus = (1300.0, 1600.0, 1900.0, 2200.0)

    specs: list[LinkSpec] = []
    for i in range(1, 9):
        specs.append(
            LinkSpec(link_id=f"NOM{i}", label="nominal", duration_s=nominal_s, seed=1000 + i)
        )

    prefixes = {
        "edfa_aging": "AGE",
        "fiber_pinch": "PIN",
        "dirty_connector": "DRT",
        "laser_drift": "DRF",
    }
    for cls_idx, (label, prefix) in enumerate(prefixes.items()):
        for j, tau in enumerate(taus, start=1):
            specs.append(
                LinkSpec(
                    link_id=f"{prefix}{j}",
                    label=label,
                    duration_s=soft_s,
                    onset_s=1800.0,
                    tau_s=tau,
                    severity=CLASS_SEVERITY[label],
                    seed=2000 + 100 * cls_idx + j,
                )
            )

    for j in range(1, 3):
        specs.append(
            LinkSpec(
                link_id=f"LOS{j}",
                label="los",
                duration_s=los_s,
                onset_s=3600.0,
                tau_s=1.0,
                seed=3000 + j,
            )
        )
    return specs


def sample_specs(duration_s: int = 1200) -> list[LinkSpec]:
    """Committed sample CSV: 3 links x 20 min (2 nominal + 1 aging link)."""
    return [
        LinkSpec(link_id="S1", label="nominal", duration_s=duration_s, seed=11),
        LinkSpec(
            link_id="S2",
            label="edfa_aging",
            duration_s=duration_s,
            onset_s=300.0,
            tau_s=400.0,
            severity=160.0,
            seed=12,
        ),
        LinkSpec(link_id="S3", label="nominal", duration_s=duration_s, seed=13),
    ]


def first_hard_fail_s(df: pd.DataFrame, link_id: str) -> float | None:
    """Seconds from stream start until the first hard failure on ``link_id``."""
    sub = df[df["link_id"] == link_id].sort_values("ts_unix_ms")
    if sub.empty:
        return None
    mask = is_hard_fail(sub["osnr_db"].to_numpy(), sub["ber"].to_numpy())
    if not mask.any():
        return None
    t0 = int(sub["ts_unix_ms"].iloc[0])
    idx = int(np.argmax(mask))
    return (int(sub["ts_unix_ms"].iloc[idx]) - t0) / 1000.0
