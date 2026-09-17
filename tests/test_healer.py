"""Self-heal policy: consecutive-window trigger, single emission, hysteresis."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.healer import main as healer_main
from apps.healer.main import Healer, Observation, Thresholds, create_app

BASE_TS_MS = 1_710_000_000_000


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"accepted": True, "message": "ok"}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def rest_calls(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - httpx kwarg name
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(healer_main.httpx, "post", fake_post)
    return calls


@pytest.fixture
def grpc_calls(monkeypatch):
    calls: list[dict] = []

    def fake_grpc(self, payload):
        calls.append(payload)
        return {"ok": True, "message": "grpc ok"}

    monkeypatch.setattr(Healer, "_emit_grpc", fake_grpc)
    return calls


@pytest.fixture
def healer() -> Healer:
    return Healer(
        thresholds=Thresholds(p_fail_threshold=0.7, consecutive_windows=3, hysteresis_s=300.0),
        topology={
            "paths": {
                "P_work_L1": {"links": ["L1"], "role": "working", "capacity": 1},
                "P_prot_L2": {"links": ["L2"], "role": "protection", "capacity": 1},
                "P_work_L3": {"links": ["L3"], "role": "working", "capacity": 1},
            },
            "protection": {"L1": "P_prot_L2", "L3": "P_prot_L2"},
        },
        rest_base="http://path_manager:8080",
        grpc_target="path_manager:8081",
    )


def observation(seconds: int, p_fail: float = 0.9, link_id: str = "L1") -> Observation:
    return Observation(
        ts_unix_ms=BASE_TS_MS + seconds * 1000,
        link_id=link_id,
        channel_id="C1",
        s_if=0.95,
        s_ae=0.85,
        p_fail_10m=p_fail,
    )


def test_three_hits_fire_exactly_one_rest_call(healer, rest_calls, grpc_calls):
    first = healer.observe(observation(0))
    second = healer.observe(observation(1))
    assert first["action"] == "counting" and first["consecutive"] == 1
    assert second["action"] == "counting" and second["consecutive"] == 2
    assert rest_calls == []

    third = healer.observe(observation(2))
    assert third["action"] == "reroute_emitted"
    assert len(rest_calls) == 1
    assert rest_calls[0]["url"] == "http://path_manager:8080/v1/reroute"

    body = rest_calls[0]["json"]
    assert body["from_path"] == "P_work_L1"
    assert body["to_path"] == "P_prot_L2"
    assert body["link_id"] == "L1"
    assert body["horizon_s"] == 600
    assert body["p_fail_10m"] == pytest.approx(0.9)
    assert body["reason"] == "soft_fail_predicted edfa_or_transceiver_aging"
    assert third["rest"]["ok"] is True


def test_reroute_is_emitted_over_both_rest_and_grpc(healer, rest_calls, grpc_calls):
    for t in range(3):
        result = healer.observe(observation(t))
    assert len(rest_calls) == 1
    assert len(grpc_calls) == 1
    assert grpc_calls[0] == rest_calls[0]["json"]
    assert result["grpc"]["ok"] is True


def test_hysteresis_suppresses_the_second_reroute(healer, rest_calls, grpc_calls):
    for t in range(3):
        healer.observe(observation(t))
    assert len(rest_calls) == 1

    # Within the 300 s hysteresis window: suppressed.
    for t in (3, 4, 5, 100, 299):
        result = healer.observe(observation(t))
        assert result["action"] == "suppressed_hysteresis"
    assert len(rest_calls) == 1

    # After hysteresis expires the healer may act again.
    result = healer.observe(observation(303))
    assert result["action"] == "reroute_emitted"
    assert len(rest_calls) == 2


def test_below_threshold_resets_the_consecutive_counter(healer, rest_calls, grpc_calls):
    healer.observe(observation(0))
    healer.observe(observation(1))
    reset = healer.observe(observation(2, p_fail=0.4))
    assert reset["action"] == "below_threshold"
    assert reset["consecutive"] == 0

    assert healer.observe(observation(3))["action"] == "counting"
    assert healer.observe(observation(4))["action"] == "counting"
    assert rest_calls == []
    assert healer.observe(observation(5))["action"] == "reroute_emitted"
    assert len(rest_calls) == 1


def test_counters_are_tracked_per_link(healer, rest_calls, grpc_calls):
    healer.observe(observation(0, link_id="L1"))
    healer.observe(observation(1, link_id="L1"))
    healer.observe(observation(2, link_id="L3"))
    assert rest_calls == []
    assert healer.observe(observation(3, link_id="L1"))["action"] == "reroute_emitted"
    assert len(rest_calls) == 1
    assert rest_calls[0]["json"]["link_id"] == "L1"


def test_link_without_protection_is_not_rerouted(healer, rest_calls, grpc_calls):
    for t in range(3):
        result = healer.observe(observation(t, link_id="L9"))
    assert result["action"] == "no_protection_path"
    assert rest_calls == []


def test_rest_failure_is_reported_but_does_not_raise(healer, grpc_calls, monkeypatch):
    def boom(url, json=None, timeout=None):  # noqa: A002
        raise RuntimeError("connection refused")

    monkeypatch.setattr(healer_main.httpx, "post", boom)
    for t in range(3):
        result = healer.observe(observation(t))
    assert result["action"] == "reroute_emitted"
    assert result["rest"]["ok"] is False
    assert "connection refused" in result["rest"]["error"]


def test_http_api_routes_observations_through_the_policy(healer, rest_calls, grpc_calls):
    client = TestClient(create_app(healer))

    assert client.get("/health").status_code == 200
    for t in range(2):
        response = client.post("/v1/observation", json=observation(t).model_dump())
        assert response.status_code == 200
        assert response.json()["action"] == "counting"

    response = client.post("/v1/observation", json=observation(2).model_dump())
    assert response.json()["action"] == "reroute_emitted"
    assert len(rest_calls) == 1

    state = client.get("/v1/state").json()
    assert state["links"]["L1"]["rerouted"] is True
    assert state["thresholds"]["consecutive_windows"] == 3


def test_thresholds_load_from_repo_config():
    thresholds = Thresholds.load()
    assert thresholds.p_fail_threshold == pytest.approx(0.7)
    assert thresholds.consecutive_windows == 3
    assert thresholds.hysteresis_s == pytest.approx(300.0)
    assert thresholds.horizon_s == 600
