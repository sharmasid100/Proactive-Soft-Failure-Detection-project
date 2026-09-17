"""Infer service: JSONL feature-frame server on :9001 + FastAPI on :8090.

Reads feature frames from the C++ ingest, scores them with the frozen
Isolation Forest / autoencoder / horizon artifacts, and POSTs each observation to
the healer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from apps.infer.serve import (
    MODELS_DIR,
    SERVICE,
    app,
    get_pipeline,
    log,
    missing_artifacts,
    score_frame,
)

DEFAULT_LISTEN_HOST = os.environ.get("INFER_LISTEN_HOST", "0.0.0.0")
DEFAULT_LISTEN_PORT = int(os.environ.get("INFER_LISTEN_PORT", "9001"))
DEFAULT_API_PORT = int(os.environ.get("INFER_API_PORT", "8090"))
DEFAULT_HEALER_URL = os.environ.get("HEALER_URL", "http://127.0.0.1:8091")

MAX_LINE_BYTES = 4 * 1024 * 1024


class InferRunner:
    """Owns the TCP frame server and the HTTP client to the healer."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        healer_url: str = DEFAULT_HEALER_URL,
        listen_host: str = DEFAULT_LISTEN_HOST,
        listen_port: int = DEFAULT_LISTEN_PORT,
    ) -> None:
        self.models_dir = models_dir
        self.healer_url = healer_url.rstrip("/")
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.frames = 0
        self.dropped = 0
        self.posted = 0
        self._client: httpx.AsyncClient | None = None

    async def post_observation(self, observation: dict) -> None:
        if self._client is None:
            return
        url = self.healer_url + "/v1/observation"
        try:
            response = await self._client.post(url, json=observation, timeout=5.0)
            self.posted += 1
            if response.status_code >= 400:
                log("healer_rejected", status_code=response.status_code, body=response.text[:400])
        except Exception as exc:
            log("healer_post_failed", url=url, error=str(exc))

    async def handle_line(self, line: str) -> dict | None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            self.dropped += 1
            log("bad_frame", error=str(exc))
            return None
        if not isinstance(frame, dict) or "seq" not in frame or "link_id" not in frame:
            self.dropped += 1
            log("bad_frame", error="missing seq/link_id")
            return None
        try:
            observation = await asyncio.to_thread(score_frame, frame, self.models_dir)
        except Exception as exc:
            self.dropped += 1
            log("score_failed", error=str(exc))
            return None

        self.frames += 1
        if self.frames % 30 == 1:
            log("scored", frames=self.frames, **observation)
        await self.post_observation(observation)
        return observation

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        log("ingest_connected", peer=str(peer))
        try:
            while True:
                try:
                    raw = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    raw = exc.partial
                    if not raw:
                        break
                except asyncio.LimitOverrunError:
                    await reader.read(MAX_LINE_BYTES)
                    self.dropped += 1
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    await self.handle_line(line)
                if not raw.endswith(b"\n"):
                    break
        finally:
            log("ingest_disconnected", peer=str(peer), frames=self.frames, dropped=self.dropped)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - peer already gone
                pass

    async def serve_frames(self) -> None:
        server = await asyncio.start_server(
            self._handle_client, self.listen_host, self.listen_port, limit=MAX_LINE_BYTES
        )
        log("frame_server_listening", host=self.listen_host, port=self.listen_port)
        async with server:
            await server.serve_forever()

    async def run(self, api_port: int | None = DEFAULT_API_PORT) -> None:
        missing = missing_artifacts(self.models_dir)
        if missing:
            raise SystemExit(
                "missing model artifacts: " + ", ".join(missing) + " (run `make train`)"
            )
        get_pipeline(self.models_dir)

        async with httpx.AsyncClient() as client:
            self._client = client
            tasks = [asyncio.create_task(self.serve_frames())]
            if api_port:
                tasks.append(asyncio.create_task(self._serve_api(api_port)))
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:  # pragma: no cover
                pass
            finally:
                for task in tasks:
                    task.cancel()

    async def _serve_api(self, port: int) -> None:
        import uvicorn

        config = uvicorn.Config(
            app, host=self.listen_host, port=port, log_level="warning", loop="asyncio"
        )
        server = uvicorn.Server(config)
        log("api_listening", host=self.listen_host, port=port)
        await server.serve()


def log_startup(**kv: Any) -> None:
    log("starting", service=SERVICE, ts=time.time(), **kv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apps.infer.main", description=__doc__)
    parser.add_argument("--host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT, help="feature-frame TCP port")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="0 disables FastAPI")
    parser.add_argument("--healer-url", default=DEFAULT_HEALER_URL)
    parser.add_argument("--models", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    runner = InferRunner(
        models_dir=Path(args.models),
        healer_url=args.healer_url,
        listen_host=args.host,
        listen_port=args.port,
    )
    log_startup(
        frame_port=args.port,
        api_port=args.api_port,
        healer_url=runner.healer_url,
        models_dir=str(runner.models_dir),
    )
    try:
        asyncio.run(runner.run(api_port=args.api_port or None))
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
