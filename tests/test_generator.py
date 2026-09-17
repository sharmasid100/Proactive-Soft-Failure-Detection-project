"""Generator physics and scenario schedules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.generator.gnpy_like import (
    BER_CEIL,
    BER_FLOOR,
    HARD_BER,
    HARD_OSNR_DB,
    LOS_RX_DBM,
    LinkSpec,
    add_horizon_labels,
    ber_from_osnr,
    demo_specs,
    first_hard_fail_s,
    generate_link,
    generate_scenario,
    is_hard_fail,
    ramp_progress,
    sample_specs,
    training_specs,
)
from ml.features import WINDOW_N


def test_ber_from_osnr_locks_formula():
    osnr = 18.5
    expected = 0.5 * math.erfc(math.sqrt(10 ** (osnr / 10) / 2.0) * 0.5)
    assert ber_from_osnr(osnr) == pytest.approx(expected, rel=1e-12)


def test_ber_from_osnr_is_monotone_decreasing_and_clipped():
    osnr = np.linspace(-5.0, 40.0, 200)
    ber = ber_from_osnr(osnr)
    assert np.all(np.diff(ber) <= 1e-18)
    assert ber.min() >= BER_FLOOR
    assert ber.max() <= BER_CEIL


def test_nominal_link_sits_at_18_5_db():
    df = generate_link(LinkSpec(link_id="N1", label="nominal", duration_s=600, seed=7))
    assert df["osnr_db"].mean() == pytest.approx(18.5, abs=0.02)
    assert df["osnr_db"].std() == pytest.approx(0.05, abs=0.02)
    assert df["laser_bias_ma"].mean() == pytest.approx(40.0, abs=0.05)
    assert df["edfa_pump_ma"].mean() == pytest.approx(180.0, abs=0.1)
    assert df["rx_power_dbm"].mean() == pytest.approx(-12.0, abs=0.05)
    assert not is_hard_fail(df["osnr_db"].to_numpy(), df["ber"].to_numpy()).any()


def test_los_collapses_osnr():
    spec = LinkSpec(link_id="X1", label="los", duration_s=120, onset_s=60.0, seed=3)
    df = generate_link(spec)
    before = df.iloc[:60]
    after = df.iloc[60:]
    assert before["osnr_db"].min() > HARD_OSNR_DB
    assert (after["osnr_db"] == 0.0).all()
    assert (after["ber"] == 0.1).all()
    assert (after["rx_power_dbm"] == LOS_RX_DBM).all()
    assert first_hard_fail_s(df, "X1") == pytest.approx(60.0)


def test_aging_reduces_osnr_and_raises_pump_over_time():
    spec = LinkSpec(
        link_id="A1",
        label="edfa_aging",
        duration_s=1200,
        onset_s=100.0,
        tau_s=300.0,
        severity=100.0,
        seed=5,
    )
    df = generate_link(spec)
    head = df.iloc[:60]
    tail = df.iloc[-60:]
    assert tail["osnr_db"].mean() < head["osnr_db"].mean() - 1.0
    assert tail["edfa_pump_ma"].mean() > head["edfa_pump_ma"].mean() + 10.0
    assert tail["ber"].mean() > head["ber"].mean()


@pytest.mark.parametrize("label", ["fiber_pinch", "dirty_connector"])
def test_loss_classes_reduce_rx_power(label):
    spec = LinkSpec(
        link_id="P1",
        label=label,
        duration_s=1200,
        onset_s=100.0,
        tau_s=300.0,
        severity=6.0,
        seed=9,
    )
    df = generate_link(spec)
    assert df["rx_power_dbm"].iloc[-1] < df["rx_power_dbm"].iloc[0] - 2.0
    assert df["osnr_db"].iloc[-1] < df["osnr_db"].iloc[0] - 1.0


def test_laser_drift_raises_bias_current():
    spec = LinkSpec(
        link_id="D1",
        label="laser_drift",
        duration_s=1200,
        onset_s=100.0,
        tau_s=300.0,
        severity=120.0,
        seed=4,
    )
    df = generate_link(spec)
    assert df["laser_bias_ma"].iloc[-1] > df["laser_bias_ma"].iloc[0] + 50.0
    assert df["osnr_db"].iloc[-1] < df["osnr_db"].iloc[0] - 1.0


def test_ramp_progress_bounds():
    assert ramp_progress(0.0, 10.0, 5.0) == pytest.approx(0.0)
    assert ramp_progress(10.0, 10.0, 5.0) == pytest.approx(0.0)
    assert 0.0 < float(ramp_progress(12.0, 10.0, 5.0)) < 1.0
    assert float(ramp_progress(1000.0, 10.0, 5.0)) == pytest.approx(1.0, abs=1e-9)


def test_demo_schedule_hits_soft_then_hard():
    """The demo must degrade softly for >=15 s after the first window, then fail hard."""
    df = generate_scenario(demo_specs())
    hard_fail_s = first_hard_fail_s(df, "L1")
    assert hard_fail_s is not None, "L1 must reach hard failure in the demo"

    # Hard failure lands after the first 60 s window exists but within 90 s.
    assert WINDOW_N <= hard_fail_s <= 90.0
    # Soft-failure window: traffic still up for >= 15 s after the first frame.
    assert hard_fail_s - (WINDOW_N - 1) >= 15.0

    l1 = df[df["link_id"] == "L1"].reset_index(drop=True)
    soft = l1.iloc[WINDOW_N - 1 : int(hard_fail_s)]
    # Degraded but not yet failed during the soft window.
    assert not is_hard_fail(soft["osnr_db"].to_numpy(), soft["ber"].to_numpy()).any()
    assert soft["osnr_db"].max() < 17.0
    assert soft["ber"].min() > l1["ber"].iloc[0] * 10

    # Protection and the second working path stay healthy.
    for link in ("L2", "L3"):
        assert first_hard_fail_s(df, link) is None


def test_training_corpus_hard_fails_between_12_and_25_minutes_after_onset():
    impaired = [s for s in training_specs(0.5) if s.label not in ("nominal", "los")]
    assert len(impaired) == 16
    for spec in impaired:
        df = generate_link(spec)
        hard_fail_s = first_hard_fail_s(df, spec.link_id)
        assert hard_fail_s is not None, f"{spec.link_id} never fails"
        minutes_after_onset = (hard_fail_s - spec.onset_s) / 60.0
        assert 12.0 <= minutes_after_onset <= 25.0, (spec.link_id, spec.label, minutes_after_onset)


def test_training_specs_shape():
    specs = training_specs(4.0)
    nominal = [s for s in specs if s.label == "nominal"]
    los = [s for s in specs if s.label == "los"]
    assert len(nominal) == 8
    assert all(s.duration_s == 4 * 3600 for s in nominal)
    assert len(los) == 2
    for label in ("edfa_aging", "fiber_pinch", "dirty_connector", "laser_drift"):
        cls = [s for s in specs if s.label == label]
        assert len(cls) == 4
        assert all(s.duration_s == 2 * 3600 and s.onset_s == 1800.0 for s in cls)
        assert all(1200.0 <= s.tau_s <= 2400.0 for s in cls)


def test_horizon_labels_flag_the_ten_minutes_before_failure():
    spec = LinkSpec(link_id="H1", label="los", duration_s=1800, onset_s=1200.0, seed=8)
    labelled = add_horizon_labels(generate_link(spec))
    assert labelled.loc[labelled.index[0], "y_horizon"] == 0
    # The step lands on index 1200, so y_horizon is 1 exactly on indices 600..1199
    # (their next 600 s contains the failure) and 0 at 599.
    assert labelled.loc[labelled.index[599], "y_horizon"] == 0
    assert labelled.loc[labelled.index[600], "y_horizon"] == 1
    assert labelled.loc[labelled.index[1199], "y_horizon"] == 1
    assert bool(labelled.loc[labelled.index[1199], "hard_fail"]) is False
    assert bool(labelled.loc[labelled.index[1200], "hard_fail"]) is True
    assert int(labelled["y_horizon"].sum()) >= 600


def test_sample_specs_cover_three_links_of_twenty_minutes():
    specs = sample_specs()
    assert [s.link_id for s in specs] == ["S1", "S2", "S3"]
    assert all(s.duration_s == 1200 for s in specs)
    df = generate_scenario(specs)
    assert len(df) == 3 * 1200
    assert df["ber"].min() > 0.0
    assert df["ber"].max() <= BER_CEIL
    # S2 degrades far enough to exercise the horizon label.
    assert first_hard_fail_s(df, "S2") is not None
    assert first_hard_fail_s(df, "S1") is None


def test_hard_failure_definition():
    assert bool(is_hard_fail(HARD_OSNR_DB - 0.1, 1e-9))
    assert bool(is_hard_fail(20.0, HARD_BER * 1.1))
    assert not bool(is_hard_fail(HARD_OSNR_DB + 0.1, HARD_BER * 0.9))
