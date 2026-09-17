"""Path manager: capacity accounting over REST and gRPC."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from apps.path_manager.main import RerouteRequest, TopologyState, create_app, load_topology

TOPOLOGY = {
    "paths": {
        "P_work_L1": {"links": ["L1"], "role": "working", "capacity": 1},
        "P_prot_L2": {"links": ["L2"], "role": "protection", "capacity": 1},
        "P_work_L3": {"links": ["L3"], "role": "working", "capacity": 1},
    },
    "protection": {"L1": "P_prot_L2", "L3": "P_prot_L2"},
}


@pytest.fixture
def state(tmp_path) -> TopologyState:
    return TopologyState(topology=TOPOLOGY, reroute_log=tmp_path / "reroutes.jsonl")


@pytest.fixture
def client(state) -> TestClient:
    return TestClient(create_app(state))


def request_body(link_id: str = "L1", from_path: str = "P_work_L1") -> dict:
    return {
        "from_path": from_path,
        "to_path": "P_prot_L2",
        "link_id": link_id,
        "reason": "soft_fail_predicted edfa_or_transceiver_aging",
        "p_fail_10m": 0.88,
        "s_if": 0.95,
        "s_ae": 0.81,
        "horizon_s": 600,
        "ts_unix_ms": 1_710_000_060_000,
    }


def test_repo_topology_matches_the_contract():
    topology = load_topology()
    assert set(topology["paths"]) == {"P_work_L1", "P_prot_L2", "P_work_L3"}
    assert topology["paths"]["P_prot_L2"]["role"] == "protection"
    assert topology["protection"] == {"L1": "P_prot_L2", "L3": "P_prot_L2"}


def test_first_reroute_l1_to_l2_is_accepted_second_is_rejected(client):
    first = client.post("/v1/reroute", json=request_body("L1"))
    assert first.status_code == 200
    assert first.json()["accepted"] is True
    assert "P_work_L1 -> P_prot_L2" in first.json()["message"]

    # Protection capacity is now 0, so L3 cannot also move onto P_prot_L2.
    second = client.post("/v1/reroute", json=request_body("L3", from_path="P_work_L3"))
    assert second.status_code == 200
    assert second.json()["accepted"] is False
    assert "capacity" in second.json()["message"]


def test_accepted_reroute_updates_topology_state(client):
    client.post("/v1/reroute", json=request_body("L1"))
    topology = client.get("/v1/topology").json()
    assert topology["paths"]["P_work_L1"]["status"] == "down"
    assert topology["paths"]["P_prot_L2"]["status"] == "active"
    assert topology["paths"]["P_prot_L2"]["capacity"] == 0
    assert len(topology["reroutes"]) == 1
    assert topology["reroutes"][0]["link_id"] == "L1"
    assert topology["reroutes"][0]["accepted"] is True


def test_accepted_reroute_is_appended_to_the_jsonl_log(state, tmp_path):
    response = state.apply_reroute(RerouteRequest(**request_body("L1")))
    assert response.accepted is True
    lines = (tmp_path / "reroutes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["link_id"] == "L1"
    assert record["to_path"] == "P_prot_L2"
    assert record["horizon_s"] == 600
    assert record["accepted"] is True


def test_repeating_the_same_move_is_idempotent(client):
    """The healer intentionally calls REST *and* gRPC; both must be accepted."""
    first = client.post("/v1/reroute", json=request_body("L1"))
    second = client.post("/v1/reroute", json=request_body("L1"))
    assert first.json()["accepted"] is True
    assert second.json()["accepted"] is True
    assert "already rides" in second.json()["message"]

    topology = client.get("/v1/topology").json()
    assert topology["paths"]["P_prot_L2"]["capacity"] == 0
    assert len(topology["reroutes"]) == 1


def test_unknown_paths_are_rejected(client):
    body = request_body("L1")
    body["to_path"] = "P_nope"
    assert client.post("/v1/reroute", json=body).json()["accepted"] is False

    body = request_body("L1")
    body["from_path"] = "P_nope"
    assert client.post("/v1/reroute", json=body).json()["accepted"] is False


def test_health_and_topology_endpoints(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    topology = client.get("/v1/topology").json()
    assert topology["protection"]["L1"] == "P_prot_L2"
    assert topology["paths"]["P_prot_L2"]["status"] == "standby"
    assert topology["reroutes"] == []


def test_state_helpers_resolve_paths(state):
    assert state.working_path_for_link("L1") == "P_work_L1"
    assert state.working_path_for_link("L2") is None
    assert state.protection_path_for_link("L1") == "P_prot_L2"
    assert state.protection_path_for_link("L9") is None


def test_grpc_servicer_shares_state_with_rest(state):
    from apps.path_manager import grpc_server

    if not grpc_server.GRPC_AVAILABLE:
        pytest.skip("generated gRPC stubs missing; run scripts/gen_protos.sh")

    from generated.optics.v1 import path_control_pb2

    service = grpc_server.PathControlService(state)
    request = path_control_pb2.RerouteRequest(**request_body("L1"))
    response = service.Reroute(request, None)
    assert response.accepted is True

    # REST sees the capacity consumed by the gRPC call.
    client = TestClient(create_app(state))
    rejected = client.post("/v1/reroute", json=request_body("L3", from_path="P_work_L3"))
    assert rejected.json()["accepted"] is False
