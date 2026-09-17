"""gRPC front end for the path manager (``optics.v1.PathControl`` on :8081).

The generated stubs live in ``generated/`` (gitignored) and are produced by
``scripts/gen_protos.sh`` / ``make proto``. When they are absent this module
degrades gracefully so the REST path still works.
"""

from __future__ import annotations

import os
import threading
from concurrent import futures
from typing import Any

from apps.path_manager.main import RerouteRequest, TopologyState, log

GRPC_AVAILABLE = True
IMPORT_ERROR: str | None = None

try:  # pragma: no cover - depends on `make proto`
    import grpc

    from generated.optics.v1 import path_control_pb2, path_control_pb2_grpc
except Exception as exc:  # pragma: no cover
    GRPC_AVAILABLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    grpc = None  # type: ignore[assignment]
    path_control_pb2 = None  # type: ignore[assignment]
    path_control_pb2_grpc = None  # type: ignore[assignment]


def _servicer_base() -> Any:
    return path_control_pb2_grpc.PathControlServicer if GRPC_AVAILABLE else object


class PathControlService(_servicer_base()):  # type: ignore[misc]
    """Applies reroutes to the shared :class:`TopologyState`."""

    def __init__(self, state: TopologyState) -> None:
        self.state = state

    def Reroute(self, request, context=None):  # noqa: N802 - gRPC naming
        parsed = RerouteRequest(
            from_path=request.from_path,
            to_path=request.to_path,
            link_id=request.link_id,
            reason=request.reason,
            p_fail_10m=request.p_fail_10m,
            s_if=request.s_if,
            s_ae=request.s_ae,
            horizon_s=request.horizon_s or 600,
            ts_unix_ms=request.ts_unix_ms,
        )
        response = self.state.apply_reroute(parsed)
        log("grpc_reroute", accepted=response.accepted, message=response.message)
        return path_control_pb2.RerouteResponse(
            accepted=response.accepted, message=response.message
        )


def build_server(state: TopologyState, host: str = "0.0.0.0", port: int = 8081):
    """Create (but do not start) the gRPC server, or ``None`` if unavailable."""
    if not GRPC_AVAILABLE:
        log("grpc_unavailable", error=IMPORT_ERROR, hint="run scripts/gen_protos.sh")
        return None
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    path_control_pb2_grpc.add_PathControlServicer_to_server(PathControlService(state), server)
    server.add_insecure_port(f"{host}:{port}")
    return server


def serve_grpc_in_thread(state: TopologyState, host: str = "0.0.0.0", port: int = 8081):
    """Start the gRPC server on a daemon thread; returns the server or ``None``."""
    server = build_server(state, host=host, port=port)
    if server is None:
        return None
    server.start()
    log("grpc_listening", host=host, port=port)
    threading.Thread(target=server.wait_for_termination, daemon=True).start()
    return server


def main() -> int:
    state = TopologyState()
    host = os.environ.get("PATH_MANAGER_HOST", "0.0.0.0")
    port = int(os.environ.get("PATH_MANAGER_GRPC_PORT", "8081"))
    server = build_server(state, host=host, port=port)
    if server is None:
        return 1
    server.start()
    log("grpc_listening", host=host, port=port)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
