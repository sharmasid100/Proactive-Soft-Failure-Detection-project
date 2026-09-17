"""Self-heal policy: threshold + hysteresis, then call the PCE over REST *and* gRPC.

FastAPI on :8091. ``POST /v1/observation`` accepts a score from the infer service;
after ``consecutive_windows`` observations at or above ``p_fail_threshold`` on the
same link, the healer resolves the protection path and emits a reroute. The
``REROUTE_EMITTED`` log line is the demo success signal.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

SERVICE = "healer"
REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = Path(os.environ.get("THRESHOLDS_PATH", REPO_ROOT / "configs" / "thresholds.yaml"))
TOPOLOGY_PATH = Path(os.environ.get("TOPOLOGY_PATH", REPO_ROOT / "configs" / "topology.yaml"))

DEFAULT_REST = os.environ.get("PATH_MANAGER_REST", "http://127.0.0.1:8080")
DEFAULT_GRPC = os.environ.get("PATH_MANAGER_GRPC", "127.0.0.1:8081")

REASON = "soft_fail_predicted edfa_or_transceiver_aging"
HORIZON_S = 600


def log(msg: str, **kv: Any) -> None:
    record = {"ts": time.time(), "service": SERVICE, "msg": msg}
    record.update(kv)
    print(json.dumps(record, default=str), flush=True)


class Observation(BaseModel):
    """Score emitted by the infer service for one feature window."""

    ts_unix_ms: int = 0
    link_id: str
    channel_id: str = "C1"
    s_if: float = 0.0
    s_ae: float = 0.0
    p_fail_10m: float = 0.0


@dataclass
class LinkState:
    consecutive: int = 0
    observations: int = 0
    last_reroute_ts: float | None = None
    rerouted: bool = False


@dataclass
class Thresholds:
    p_fail_threshold: float = 0.7
    consecutive_windows: int = 3
    hysteresis_s: float = 300.0
    horizon_s: int = HORIZON_S

    @classmethod
    def load(cls, path: Path = THRESHOLDS_PATH) -> "Thresholds":
        try:
            blob = yaml.safe_load(path.read_text()) or {}
        except OSError:
            return cls()
        return cls(
            p_fail_threshold=float(blob.get("p_fail_threshold", 0.7)),
            consecutive_windows=int(blob.get("consecutive_windows", 3)),
            hysteresis_s=float(blob.get("hysteresis_s", 300)),
            horizon_s=int(blob.get("horizon_s", HORIZON_S)),
        )


def load_topology(path: Path = TOPOLOGY_PATH) -> dict:
    return yaml.safe_load(path.read_text()) or {}


@dataclass
class Healer:
    """Threshold + hysteresis state machine with REST and gRPC egress."""

    thresholds: Thresholds = field(default_factory=Thresholds.load)
    topology: dict = field(default_factory=load_topology)
    rest_base: str = DEFAULT_REST
    grpc_target: str = DEFAULT_GRPC
    clock: Callable[[], float] = time.time
    links: dict[str, LinkState] = field(default_factory=dict)

    # --- topology --------------------------------------------------------- #
    def working_path_for_link(self, link_id: str) -> str | None:
        for name, spec in (self.topology.get("paths") or {}).items():
            if spec.get("role") == "working" and link_id in (spec.get("links") or []):
                return name
        return None

    def protection_path_for_link(self, link_id: str) -> str | None:
        return (self.topology.get("protection") or {}).get(link_id)

    # --- policy ----------------------------------------------------------- #
    def _now(self, observation: Observation) -> float:
        if observation.ts_unix_ms:
            return observation.ts_unix_ms / 1000.0
        return float(self.clock())

    def observe(self, observation: Observation) -> dict:
        state = self.links.setdefault(observation.link_id, LinkState())
        state.observations += 1
        now = self._now(observation)

        if observation.p_fail_10m < self.thresholds.p_fail_threshold:
            state.consecutive = 0
            return {"action": "below_threshold", "link_id": observation.link_id, "consecutive": 0}

        state.consecutive += 1
        if state.consecutive < self.thresholds.consecutive_windows:
            return {
                "action": "counting",
                "link_id": observation.link_id,
                "consecutive": state.consecutive,
            }

        if (
            state.last_reroute_ts is not None
            and now - state.last_reroute_ts < self.thresholds.hysteresis_s
        ):
            log(
                "reroute_suppressed",
                reason="hysteresis",
                link_id=observation.link_id,
                since_last_s=round(now - state.last_reroute_ts, 3),
            )
            return {
                "action": "suppressed_hysteresis",
                "link_id": observation.link_id,
                "consecutive": state.consecutive,
            }

        from_path = self.working_path_for_link(observation.link_id)
        to_path = self.protection_path_for_link(observation.link_id)
        if from_path is None or to_path is None:
            log("no_protection_path", link_id=observation.link_id, from_path=from_path, to_path=to_path)
            return {"action": "no_protection_path", "link_id": observation.link_id}

        payload = {
            "from_path": from_path,
            "to_path": to_path,
            "link_id": observation.link_id,
            "reason": REASON,
            "p_fail_10m": float(observation.p_fail_10m),
            "s_if": float(observation.s_if),
            "s_ae": float(observation.s_ae),
            "horizon_s": int(self.thresholds.horizon_s),
            "ts_unix_ms": int(observation.ts_unix_ms or self.clock() * 1000),
        }

        rest_result = self._emit_rest(payload)
        grpc_result = self._emit_grpc(payload)

        state.last_reroute_ts = now
        state.rerouted = True

        log("REROUTE_EMITTED", rest=rest_result, grpc=grpc_result, **payload)
        return {
            "action": "reroute_emitted",
            "link_id": observation.link_id,
            "request": payload,
            "rest": rest_result,
            "grpc": grpc_result,
        }

    # --- egress ----------------------------------------------------------- #
    def _emit_rest(self, payload: dict) -> dict:
        url = self.rest_base.rstrip("/") + "/v1/reroute"
        try:
            response = httpx.post(url, json=payload, timeout=5.0)
            body: Any
            try:
                body = response.json()
            except ValueError:
                body = response.text
            return {"ok": response.status_code < 400, "status_code": response.status_code, "body": body}
        except Exception as exc:
            log("rest_emit_failed", url=url, error=str(exc))
            return {"ok": False, "error": str(exc)}

    def _emit_grpc(self, payload: dict) -> dict:
        try:
            import grpc

            from generated.optics.v1 import path_control_pb2, path_control_pb2_grpc
        except Exception as exc:
            log("grpc_stubs_unavailable", error=str(exc), hint="run scripts/gen_protos.sh")
            return {"ok": False, "error": f"stubs unavailable: {exc}"}

        try:
            with grpc.insecure_channel(self.grpc_target) as channel:
                stub = path_control_pb2_grpc.PathControlStub(channel)
                request = path_control_pb2.RerouteRequest(**payload)
                response = stub.Reroute(request, timeout=5.0)
                return {"ok": bool(response.accepted), "message": response.message}
        except Exception as exc:
            log("grpc_emit_failed", target=self.grpc_target, error=str(exc))
            return {"ok": False, "error": str(exc)}


def create_app(healer: Healer | None = None) -> FastAPI:
    app = FastAPI(title="optics-softfail healer", version="0.1.0")
    app.state.healer = healer if healer is not None else Healer()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": SERVICE, "links": len(app.state.healer.links)}

    @app.get("/v1/state")
    def state() -> dict:
        return {
            "thresholds": app.state.healer.thresholds.__dict__,
            "links": {k: v.__dict__ for k, v in app.state.healer.links.items()},
        }

    @app.post("/v1/observation")
    def observation(obs: Observation) -> dict:
        return app.state.healer.observe(obs)

    return app


app = create_app()


def main() -> int:
    import uvicorn

    host = os.environ.get("HEALER_HOST", "0.0.0.0")
    port = int(os.environ.get("HEALER_PORT", "8091"))
    log(
        "starting",
        host=host,
        port=port,
        path_manager_rest=app.state.healer.rest_base,
        path_manager_grpc=app.state.healer.grpc_target,
        thresholds=app.state.healer.thresholds.__dict__,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
