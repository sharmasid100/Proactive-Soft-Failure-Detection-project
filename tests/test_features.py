"""Feature-frame construction and window statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generator.gnpy_like import LinkSpec, generate_link
from ml import features as F


def _ramp_samples(n: int = F.WINDOW_N) -> list[dict]:
    return [
        {
            "ts_unix_ms": 1_710_000_000_000 + i * 1000,
            "link_id": "L1",
            "channel_id": "C1",
            "osnr_db": 10.0 + 0.1 * i,
            "ber": 10 ** (-9.0 + 0.01 * i),
            "laser_bias_ma": 40.0 + 0.25 * i,
            "edfa_pump_ma": 180.0 + 0.5 * i,
            "rx_power_dbm": -12.0 - 0.05 * i,
        }
        for i in range(n)
    ]


def test_ols_slope_on_linear_ramp_is_positive_and_exact():
    values = [3.0 + 2.0 * i for i in range(F.WINDOW_N)]
    assert F.ols_slope(values) == pytest.approx(2.0)
    assert F.ols_slope(values) > 0
    assert F.ols_slope(list(reversed(values))) == pytest.approx(-2.0)
    assert F.ols_slope([5.0] * F.WINDOW_N) == pytest.approx(0.0)
    assert F.ols_slope([1.0]) == 0.0


def test_mean_and_slope_match_numpy_on_index_ramp():
    values = list(range(F.WINDOW_N))
    assert np.mean(values) == pytest.approx(29.5)
    assert F.ols_slope(values) == pytest.approx(1.0)


def test_feature_vector_has_eight_entries_in_fixed_order():
    assert len(F.FEATURE_NAMES) == 8
    assert F.FEATURE_NAMES == (
        "osnr_mean",
        "osnr_std",
        "osnr_slope",
        "log10_ber_mean",
        "log10_ber_std",
        "log10_ber_slope",
        "laser_bias_mean",
        "laser_bias_slope",
    )
    frame = F.make_frame(_ramp_samples())
    vector = F.stats_vector(frame["stats"])
    assert vector.shape == (8,)
    assert np.all(np.isfinite(vector))


def test_frame_shape_and_channels():
    frame = F.make_frame(_ramp_samples())
    assert frame["n"] == F.WINDOW_N
    assert frame["link_id"] == "L1"
    assert frame["channel_id"] == "C1"
    assert set(frame["seq"]) == set(F.SEQ_CHANNELS)
    for channel in F.SEQ_CHANNELS:
        assert len(frame["seq"][channel]) == F.WINDOW_N
    matrix = F.frame_matrix(frame)
    assert matrix.shape == (F.WINDOW_N, 5)


def test_frame_slopes_are_positive_on_a_rising_ramp():
    stats = F.make_frame(_ramp_samples())["stats"]
    assert stats["osnr_slope"] == pytest.approx(0.1)
    assert stats["laser_bias_slope"] == pytest.approx(0.25)
    assert stats["log10_ber_slope"] == pytest.approx(0.01)
    assert stats["osnr_mean"] == pytest.approx(10.0 + 0.1 * 29.5)


def test_make_frame_rejects_wrong_window_length():
    with pytest.raises(ValueError):
        F.make_frame(_ramp_samples(F.WINDOW_N - 1))


def test_log10_ber_floors_at_1e_15():
    assert F.log10_ber(1e-6) == pytest.approx(-6.0)
    assert F.log10_ber(0.0) == pytest.approx(-15.0)
    assert F.log10_ber(1e-30) == pytest.approx(-15.0)


def test_optional_fields_get_ingest_defaults():
    samples = _ramp_samples()
    for sample in samples:
        sample["edfa_pump_ma"] = None
        sample.pop("rx_power_dbm")
    seq = F.sequences_from_samples(samples)
    assert seq["edfa_pump_ma"] == [F.EDFA_PUMP_DEFAULT] * F.WINDOW_N
    assert seq["rx_power_dbm"] == [F.RX_POWER_DEFAULT] * F.WINDOW_N


def test_windows_from_dataframe_matches_make_frame():
    df = generate_link(LinkSpec(link_id="W1", label="nominal", duration_s=200, seed=2))
    x_stats, x_seq, meta = F.windows_from_dataframe(df, stride=1)
    assert len(meta) == 200 - F.WINDOW_N + 1
    assert x_stats.shape == (len(meta), 8)
    assert x_seq.shape == (len(meta), F.WINDOW_N, 5)

    frame = F.make_frame(df.iloc[: F.WINDOW_N].to_dict(orient="records"))
    np.testing.assert_allclose(F.frame_matrix(frame), x_seq[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F.stats_vector(frame["stats"]), x_stats[0], rtol=1e-9, atol=1e-12)
    assert int(meta["ts_unix_ms"].iloc[0]) == int(df["ts_unix_ms"].iloc[F.WINDOW_N - 1])


def test_windows_from_dataframe_drops_incomplete_windows():
    df = generate_link(LinkSpec(link_id="S1", label="nominal", duration_s=30, seed=1))
    x_stats, x_seq, meta = F.windows_from_dataframe(df, stride=1)
    assert len(meta) == 0
    assert x_stats.shape == (0, 8)


def test_windows_respect_stride_and_group_by_link():
    a = generate_link(LinkSpec(link_id="A", label="nominal", duration_s=120, seed=1))
    b = generate_link(LinkSpec(link_id="B", label="nominal", duration_s=120, seed=2))
    df = pd.concat([a, b], ignore_index=True)
    _, x_seq, meta = F.windows_from_dataframe(df, stride=10)
    per_link = (120 - F.WINDOW_N) // 10 + 1
    assert len(meta) == 2 * per_link
    assert set(meta["link_id"]) == {"A", "B"}
    assert x_seq.shape[0] == len(meta)


def test_iter_frames_yields_one_frame_per_sample_once_full():
    samples = _ramp_samples(65)
    frames = list(F.iter_frames(samples))
    assert len(frames) == 65 - F.WINDOW_N + 1
    assert frames[0]["ts_unix_ms"] == samples[F.WINDOW_N - 1]["ts_unix_ms"]
    assert frames[-1]["ts_unix_ms"] == samples[-1]["ts_unix_ms"]


def test_stats_from_sequences_matches_scalar_path():
    df = generate_link(LinkSpec(link_id="C", label="edfa_aging", duration_s=200,
                               onset_s=10.0, tau_s=50.0, severity=60.0, seed=3))
    x_stats, x_seq, _ = F.windows_from_dataframe(df, stride=7)
    recomputed = F.stats_from_sequences(x_seq)
    np.testing.assert_allclose(x_stats, recomputed, rtol=1e-12, atol=1e-12)
