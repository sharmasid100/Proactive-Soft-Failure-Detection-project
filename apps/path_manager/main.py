"""Path manager: stub PCE that applies reroutes to an in-memory topology.

REST on :8080 (``POST /v1/reroute``, ``GET /v1/topology``, ``GET /health``); the
same logic is exposed over gRPC on :8081 by :mod:`apps.path_manager.grpc_server`.
Both share one :class:`TopologyState` instance when started from ``__main__``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

SERVICE = "path_manager"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_PATH = Path(os.environ.get("TOPOLOGY_PATH", REPO_ROOT / "configs" / "topology.yaml"))
REROUTE_LOG = Path(os.environ.get("REROUTE_LOG", REPO_ROOT / "data" / "synthetic" / "reroutes.jsonl"))
HORIZON_S = 600


def log(msg: str, **kv: Any) -> None:
    record = {"ts": time.time(), "service": SERVICE, "msg": msg}
    record.update(kv)
    print(json.dumps(record, default=str), flush=True)


class RerouteRequest(BaseModel):
    """JSON mirror of ``optics.v1.RerouteRequest``."""

    from_path: str
    to_path: str
    link_id: str
    reason: str = ""
    p_fail_10m: float = 0.0
    s_if: float = 0.0
    s_ae: float = 0.0
    horizon_s: int = HORIZON_S
    ts_unix_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class RerouteResponse(BaseModel):
    """JSON mirror of ``optics.v1.RerouteResponse``."""

    accepted: bool
    message: str


def load_topology(path: Path = TOPOLOGY_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TopologyState:
    """In-memory lightpath state: capacity, status, and accepted reroutes."""

    def __init__(self, topology: dict | None = None, reroute_log: Path | None = None) -> None:
        topology = topology if topology is not None else load_topology()
        self._lock = threading.Lock()
        self.protection: dict[str, str] = dict(topology.get("protection") or {})
        self.paths: dict[str, dict] = {}
        for name, spec in (topology.get("paths") or {}).items():
            self.paths[name] = {
                "links": list(spec.get("links") or []),
                "role": str(spec.get("role", "working")),
                "capacity": int(spec.get("capacity", 1)),
                "capacity_total": int(spec.get("capacity", 1)),
                "status": "standby" if spec.get("role") == "protection" else "active",
            }
        self.reroutes: list[dict] = []
        # link_id -> path it currently rides, so a repeated request for the same
        # move (the healer deliberately calls both REST and gRPC) is idempotent
        # instead of consuming protection capacity twice.
        self.moved: dict[str, str] = {}
        self.reroute_log = reroute_log if reroute_log is not None else REROUTE_LOG

    # --- queries ---------------------------------------------------------- #
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "paths": {name: dict(spec) for name, spec in self.paths.items()},
                "protection": dict(self.protection),
                "reroutes": list(self.reroutes),
            }

    def working_path_for_link(self, link_id: str) -> str | None:
        for name, spec in self.paths.items():
            if spec["role"] == "working" and link_id in spec["links"]:
                return name
        return None

    def protection_path_for_link(self, link_id: str) -> str | None:
        return self.protection.get(link_id)

    # --- mutation --------------------------------------------------------- #
    def apply_reroute(self, request: RerouteRequest) -> RerouteResponse:
        with self._lock:
            if request.from_path not in self.paths:
                return RerouteResponse(accepted=False, message=f"unknown from_path {request.from_path}")
            if request.to_path not in self.paths:
                return RerouteResponse(accepted=False, message=f"unknown to_path {request.to_path}")
            if self.moved.get(request.link_id) == request.to_path:
                log(
                    "reroute_already_applied",
                    from_path=request.from_path,
                    to_path=request.to_path,
                    link_id=request.link_id,
                )
                return RerouteResponse(
                    accepted=True,
                    message=f"{request.link_id} already rides {request.to_path}",
                )
            target = self.paths[request.to_path]
            if target["capacity"] <= 0:
                log(
                    "reroute_rejected",
                    reason="no_capacity",
                    from_path=request.from_path,
                    to_path=request.to_path,
                    link_id=request.link_id,
                )
                return RerouteResponse(
                    accepted=False,
                    message=f"{request.to_path} has no spare capacity",
                )

            target["capacity"] -= 1
            target["status"] = "active"
            self.paths[request.from_path]["status"] = "down"
            self.moved[request.link_id] = request.to_path
            record = request.model_dump()
            record["accepted"] = True
            self.reroutes.append(record)

        self._append_log(record)
        log("reroute_accepted", **record)
        return RerouteResponse(
            accepted=True,
            message=f"traffic moved {request.from_path} -> {request.to_path}",
        )

    def _append_log(self, record: dict) -> None:
        try:
            self.reroute_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.reroute_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # stdout is always the source of truth
            log("reroute_log_write_failed", error=str(exc), file=str(self.reroute_log))


def create_app(state: TopologyState | None = None) -> FastAPI:
    app = FastAPI(title="optics-softfail path manager", version="0.1.0")
    app.state.topology = state if state is not None else TopologyState()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": SERVICE, "paths": len(app.state.topology.paths)}

    @app.get("/v1/topology")
    def topology() -> dict:
        return app.state.topology.snapshot()

    @app.post("/v1/reroute", response_model=RerouteResponse)
    def reroute(request: RerouteRequest) -> RerouteResponse:
        return app.state.topology.apply_reroute(request)

    return app


app = create_app()


def main() -> int:
    import uvicorn

    from apps.path_manager.grpc_server import serve_grpc_in_thread

    host = os.environ.get("PATH_MANAGER_HOST", "0.0.0.0")
    rest_port = int(os.environ.get("PATH_MANAGER_REST_PORT", "8080"))
    grpc_port = int(os.environ.get("PATH_MANAGER_GRPC_PORT", "8081"))

    server = serve_grpc_in_thread(app.state.topology, host=host, port=grpc_port)
    log("starting", rest_port=rest_port, grpc_port=grpc_port, grpc=bool(server))
    uvicorn.run(app, host=host, port=rest_port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
