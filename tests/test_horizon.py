"""Horizon head: calibration gate on nominal traffic and separation from aging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generator.gnpy_like import LinkSpec, generate_link, generate_scenario
from ml import features as F
from ml.models.horizon import (
    HORIZON_S,
    P_FAIL_THRESHOLD,
    HorizonHead,
    ScoringPipeline,
    combine_scores,
    false_positive_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

MIN_NOMINAL_WINDOWS = 500
MAX_FPR = 0.02

pytestmark = pytest.mark.skipif(
    not ScoringPipeline.artifacts_present(MODELS_DIR),
    reason="model artifacts missing; run `make train`",
)


@pytest.fixture(scope="module")
def pipeline() -> ScoringPipeline:
    return ScoringPipeline.load(MODELS_DIR)


@pytest.fixture(scope="module")
def nominal_windows():
    """>=500 nominal windows, generated in-test with unseen seeds."""
    specs = [
        LinkSpec(link_id=f"FPR{i}", label="nominal", duration_s=700, seed=90_000 + i)
        for i in range(1, 4)
    ]
    df = generate_scenario(specs)
    x_stats, x_seq, meta = F.windows_from_dataframe(df, stride=1)
    assert len(meta) >= MIN_NOMINAL_WINDOWS, f"fixture too small: {len(meta)}"
    return x_stats, x_seq, meta


@pytest.fixture(scope="module")
def aged_windows():
    spec = LinkSpec(
        link_id="AGED",
        label="edfa_aging",
        duration_s=2400,
        onset_s=300.0,
        tau_s=900.0,
        severity=150.0,
        seed=91_001,
    )
    df = generate_link(spec)
    return F.windows_from_dataframe(df, stride=5)


def test_artifacts_exist():
    for path in ScoringPipeline.required_artifacts(MODELS_DIR):
        assert path.exists(), path


def test_horizon_head_is_a_logistic_of_the_combined_score():
    head = HorizonHead(k=10.0, s0=0.5)
    assert head.predict_one(0.5) == pytest.approx(0.5)
    assert head.predict_one(1.0) > head.predict_one(0.9) > head.predict_one(0.5)
    assert 0.0 < head.predict_one(0.0) < 0.5
    assert head.horizon_s == HORIZON_S


def test_combine_scores_is_the_even_average():
    assert combine_scores(0.2, 0.8) == pytest.approx(0.5)
    np.testing.assert_allclose(combine_scores([0.0, 1.0], [1.0, 1.0]), [0.5, 1.0])


def test_horizon_head_fit_recovers_a_separating_boundary():
    rng = np.random.default_rng(0)
    s_neg = rng.uniform(0.0, 0.4, 400)
    s_pos = rng.uniform(0.6, 1.0, 400)
    s = np.concatenate([s_neg, s_pos])
    y = np.concatenate([np.zeros(400, int), np.ones(400, int)])
    head = HorizonHead.fit(s, y)
    assert head.k > 0
    assert 0.3 < head.s0 < 0.7
    assert head.predict_one(0.9) > 0.7
    assert head.predict_one(0.1) < 0.3


def test_horizon_head_roundtrip(tmp_path):
    head = HorizonHead(k=7.5, s0=0.62)
    path = head.save(tmp_path)
    assert path.name == "horizon.json"
    loaded = HorizonHead.load(tmp_path)
    assert loaded.k == pytest.approx(7.5)
    assert loaded.s0 == pytest.approx(0.62)


def test_false_positive_rate_on_nominal_holdout_is_at_most_two_percent(pipeline, nominal_windows):
    x_stats, x_seq, meta = nominal_windows
    scores = pipeline.score_batch(x_stats, x_seq)
    fpr = false_positive_rate(scores["p_fail_10m"], P_FAIL_THRESHOLD)
    assert len(meta) >= MIN_NOMINAL_WINDOWS
    assert fpr <= MAX_FPR, f"FPR {fpr:.4f} exceeds {MAX_FPR} on {len(meta)} nominal windows"


def test_aged_windows_score_higher_than_nominal(pipeline, nominal_windows, aged_windows):
    nom_stats, nom_seq, _ = nominal_windows
    aged_stats, aged_seq, aged_meta = aged_windows
    nominal_p = pipeline.score_batch(nom_stats, nom_seq)["p_fail_10m"]
    aged_p = pipeline.score_batch(aged_stats, aged_seq)["p_fail_10m"]

    assert aged_p.mean() > nominal_p.mean()
    # The tail of the aging trace (close to hard failure) must clear the threshold.
    late = aged_p[-20:]
    assert late.mean() > P_FAIL_THRESHOLD
    # Aged windows cross the operating threshold far more often than nominal ones.
    aged_alarm_rate = float(np.mean(aged_p >= P_FAIL_THRESHOLD))
    nominal_alarm_rate = float(np.mean(nominal_p >= P_FAIL_THRESHOLD))
    assert aged_alarm_rate > 0.2
    assert aged_alarm_rate > nominal_alarm_rate
    assert np.median(aged_p) > np.quantile(nominal_p, 0.99)
    assert len(aged_meta) > 0


def test_score_frame_matches_batch_scoring(pipeline):
    df = generate_link(
        LinkSpec(link_id="F1", label="nominal", duration_s=F.WINDOW_N, seed=99_001)
    )
    frame = F.make_frame(df.to_dict(orient="records"))
    single = pipeline.score_frame(frame)
    batch = pipeline.score_batch(
        F.frame_stats_vector(frame).reshape(1, -1), F.frame_matrix(frame)[None, :, :]
    )
    assert single["s_if"] == pytest.approx(float(batch["s_if"][0]))
    assert single["s_ae"] == pytest.approx(float(batch["s_ae"][0]))
    assert single["p_fail_10m"] == pytest.approx(float(batch["p_fail_10m"][0]))
    assert 0.0 <= single["p_fail_10m"] <= 1.0


def test_scores_are_bounded(pipeline, aged_windows):
    aged_stats, aged_seq, _ = aged_windows
    scores = pipeline.score_batch(aged_stats, aged_seq)
    for key in ("s_if", "s_ae", "p_fail_10m"):
        assert scores[key].min() >= 0.0
        assert scores[key].max() <= 1.0


def test_demo_trace_fires_before_hard_failure(pipeline):
    from data.generator.gnpy_like import demo_specs, first_hard_fail_s

    df = generate_scenario(demo_specs())
    hard_fail_s = first_hard_fail_s(df, "L1")
    l1 = df[df["link_id"] == "L1"].reset_index(drop=True)
    x_stats, x_seq, meta = F.windows_from_dataframe(l1, stride=1)
    p_fail = pipeline.score_batch(x_stats, x_seq)["p_fail_10m"]

    t0 = int(l1["ts_unix_ms"].iloc[0])
    offsets = (meta["ts_unix_ms"].to_numpy() - t0) // 1000
    hits = [int(offsets[i]) for i in range(len(p_fail)) if p_fail[i] >= P_FAIL_THRESHOLD]
    assert hits, "demo trace never crosses the operating threshold"
    # Three consecutive windows must clear the threshold before hard failure.
    assert len(hits) >= 3
    assert hits[2] <= hard_fail_s

    protection = df[df["link_id"] == "L2"].reset_index(drop=True)
    p_stats, p_seq, _ = F.windows_from_dataframe(protection, stride=1)
    p_protection = pipeline.score_batch(p_stats, p_seq)["p_fail_10m"]
    assert false_positive_rate(p_protection, P_FAIL_THRESHOLD) <= MAX_FPR


def test_sample_csv_is_committed_and_scoreable(pipeline):
    csv = REPO_ROOT / "data" / "sample" / "sample_links.csv"
    assert csv.exists(), "data/sample/sample_links.csv must be committed"
    df = pd.read_csv(csv)
    assert df["link_id"].nunique() == 3
    assert len(df) >= 3 * 20 * 60
    assert csv.stat().st_size <= 2 * 1024 * 1024
    x_stats, x_seq, meta = F.windows_from_dataframe(df.head(3 * 200), stride=20)
    scores = pipeline.score_batch(x_stats, x_seq)
    assert len(scores["p_fail_10m"]) == len(meta) > 0
