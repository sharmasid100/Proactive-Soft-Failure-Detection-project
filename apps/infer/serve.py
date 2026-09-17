"""FastAPI surface for the infer service (:8090) and the frozen model loader.

``GET /health`` reports 200 only when every model artifact exists;
``POST /v1/score`` scores one feature frame (used by tests and manual probing).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.models.horizon import HORIZON_S, ScoringPipeline

SERVICE = "infer"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))

_PIPELINE: ScoringPipeline | None = None


def log(msg: str, **kv: Any) -> None:
    record = {"ts": time.time(), "service": SERVICE, "msg": msg}
    record.update(kv)
    print(json.dumps(record, default=str), flush=True)


class FeatureFrame(BaseModel):
    """The JSONL feature frame emitted by the C++ ingest (CLAUDE.md 4.2)."""

    ts_unix_ms: int = 0
    link_id: str
    channel_id: str = "C1"
    n: int = 60
    seq: dict[str, list[float]]
    stats: dict[str, float] | None = None


class ScoreResponse(BaseModel):
    ts_unix_ms: int
    link_id: str
    channel_id: str
    s_if: float
    s_ae: float
    p_fail_10m: float
    horizon_s: int = Field(default=HORIZON_S)


def artifacts_present(models_dir: Path = MODELS_DIR) -> bool:
    return ScoringPipeline.artifacts_present(models_dir)


def missing_artifacts(models_dir: Path = MODELS_DIR) -> list[str]:
    return [str(p) for p in ScoringPipeline.required_artifacts(models_dir) if not p.exists()]


def get_pipeline(models_dir: Path = MODELS_DIR, reload: bool = False) -> ScoringPipeline:
    """Load (and cache) the frozen scoring pipeline."""
    global _PIPELINE
    if _PIPELINE is None or reload:
        _PIPELINE = ScoringPipeline.load(models_dir)
        log("models_loaded", models_dir=str(models_dir))
    return _PIPELINE


def score_frame(frame: dict, models_dir: Path = MODELS_DIR) -> dict:
    """Score one feature frame into an observation payload for the healer."""
    scores = get_pipeline(models_dir).score_frame(frame)
    return {
        "ts_unix_ms": int(frame.get("ts_unix_ms", 0)),
        "link_id": str(frame.get("link_id", "")),
        "channel_id": str(frame.get("channel_id", "C1")),
        "s_if": scores["s_if"],
        "s_ae": scores["s_ae"],
        "p_fail_10m": scores["p_fail_10m"],
    }


def create_app(models_dir: Path = MODELS_DIR) -> FastAPI:
    app = FastAPI(title="optics-softfail infer", version="0.1.0")
    app.state.models_dir = models_dir

    @app.get("/health")
    def health():
        missing = missing_artifacts(app.state.models_dir)
        if missing:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "service": SERVICE, "missing": missing},
            )
        return {"status": "ok", "service": SERVICE, "models_dir": str(app.state.models_dir)}

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(frame: FeatureFrame):
        observation = score_frame(frame.model_dump(), app.state.models_dir)
        return ScoreResponse(horizon_s=HORIZON_S, **observation)

    return app


app = create_app()
