"""Telemetry producer: synthetic generator, CSV replay, and JSONL TCP streamer.

Usage::

    python -m apps.producer.main generate --out data/synthetic --hours 4
    python -m apps.producer.main stream --host 127.0.0.1 --port 9000 --demo
    python -m apps.producer.main replay --csv data/sample/sample_links.csv \
        --host 127.0.0.1 --port 9000 --rate 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import pandas as pd

from data.generator.gnpy_like import (
    RAW_COLUMNS,
    LinkSpec,
    add_horizon_labels,
    demo_specs,
    generate_scenario,
    sample_specs,
    training_specs,
)

SERVICE = "producer"
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "sample_links.csv"


def log(msg: str, **kv) -> None:
    record = {"ts": time.time(), "service": SERVICE, "msg": msg}
    record.update(kv)
    print(json.dumps(record, default=str), flush=True)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}.parquet"
    try:
        df.to_parquet(target, index=False)
        return target
    except Exception as exc:  # pyarrow missing / unusable -> CSV fallback
        log("parquet_unavailable_fallback_csv", error=str(exc))
        target = out_dir / f"{stem}.csv"
        df.to_csv(target, index=False)
        return target


def write_sample_csv(path: Path = SAMPLE_CSV, duration_s: int = 1200) -> Path:
    """(Re)write the committed sample CSV: 3 links x 20 min @ 1 Hz."""
    df = generate_scenario(sample_specs(duration_s))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def cmd_generate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    specs = training_specs(hours=args.hours)
    buckets: dict[str, list[LinkSpec]] = {"nominal": [], "soft_fail": [], "los": []}
    for spec in specs:
        if spec.label == "nominal":
            buckets["nominal"].append(spec)
        elif spec.label == "los":
            buckets["los"].append(spec)
        else:
            buckets["soft_fail"].append(spec)

    written = []
    for stem, group in buckets.items():
        if not group:
            continue
        df = add_horizon_labels(generate_scenario(group))
        target = _write_table(df, out_dir, stem)
        written.append(str(target))
        log("wrote", file=str(target), rows=int(len(df)), links=len(group))

    if not args.no_sample:
        sample = write_sample_csv()
        written.append(str(sample))
        log("wrote_sample", file=str(sample))

    log("generate_done", files=written)
    return 0


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def _clean_record(rec: dict, keep_label: bool) -> dict:
    out = {}
    for key in RAW_COLUMNS:
        if key == "label" and not keep_label:
            continue
        if key not in rec:
            continue
        val = rec[key]
        if key in ("link_id", "channel_id", "label"):
            out[key] = str(val)
        elif key == "ts_unix_ms":
            out[key] = int(val)
        else:
            fval = float(val)
            if not math.isfinite(fval):
                continue
            out[key] = fval
    return out


def _connect(host: str, port: int, timeout_s: float = 30.0) -> socket.socket:
    deadline = time.time() + timeout_s
    delay = 0.5
    last: Exception | None = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            log("connected", host=host, port=port)
            return sock
        except OSError as exc:  # ingest not up yet
            last = exc
            log("connect_retry", host=host, port=port, error=str(exc))
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    raise SystemExit(f"could not connect to {host}:{port}: {last}")


def stream_frames(
    frames: list[dict],
    host: str,
    port: int,
    rate: float = 1.0,
    keep_label: bool = False,
    loop: bool = False,
) -> int:
    """Send raw samples as JSONL over TCP, paced at ``rate`` samples/second/link."""
    sock = _connect(host, port)
    sent = 0
    interval = 0.0 if rate <= 0 else 1.0 / rate
    try:
        while True:
            prev_ts: int | None = None
            for rec in frames:
                clean = _clean_record(rec, keep_label)
                if prev_ts is not None and interval > 0.0:
                    if int(clean["ts_unix_ms"]) != prev_ts:
                        time.sleep(interval)
                prev_ts = int(clean["ts_unix_ms"])
                payload = (json.dumps(clean) + "\n").encode("utf-8")
                sock.sendall(payload)
                sent += 1
                if sent % 100 == 0:
                    log("sent", samples=sent)
            if not loop:
                break
    except (BrokenPipeError, ConnectionResetError) as exc:
        log("downstream_closed", error=str(exc), samples=sent)
    finally:
        sock.close()
    log("stream_done", samples=sent)
    return sent


def cmd_stream(args: argparse.Namespace) -> int:
    demo = args.demo or os.environ.get("OPTICS_DEMO") == "1"
    specs = demo_specs() if demo else demo_specs(duration_s=args.duration)
    df = generate_scenario(specs, start_ts_ms=int(time.time() * 1000))
    log("stream_start", demo=demo, links=sorted(df["link_id"].unique().tolist()), rows=int(len(df)))
    records = df.to_dict(orient="records")
    stream_frames(records, args.host, args.port, rate=args.rate, keep_label=False, loop=args.loop)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        log("csv_missing", file=str(csv_path))
        return 1
    df = pd.read_csv(csv_path)
    if args.retime:
        offset = int(time.time() * 1000) - int(df["ts_unix_ms"].min())
        df["ts_unix_ms"] = df["ts_unix_ms"] + offset
    df = df.sort_values(["ts_unix_ms", "link_id"], kind="stable")
    log("replay_start", file=str(csv_path), rows=int(len(df)))
    stream_frames(
        df.to_dict(orient="records"),
        args.host,
        args.port,
        rate=args.rate,
        keep_label=False,
        loop=args.loop,
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apps.producer.main", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the synthetic training corpus")
    gen.add_argument("--out", default="data/synthetic")
    gen.add_argument("--hours", type=float, default=4.0)
    gen.add_argument("--no-sample", action="store_true", help="do not refresh sample CSV")
    gen.set_defaults(func=cmd_generate)

    stream = sub.add_parser("stream", help="stream generated telemetry to ingest")
    stream.add_argument("--host", default=os.environ.get("INGEST_HOST", "127.0.0.1"))
    stream.add_argument("--port", type=int, default=int(os.environ.get("INGEST_PORT", "9000")))
    stream.add_argument("--rate", type=float, default=1.0, help="samples/s per link (0 = as fast as possible)")
    stream.add_argument("--demo", action="store_true", help="compressed demo onsets")
    stream.add_argument("--duration", type=int, default=1800)
    stream.add_argument("--loop", action="store_true")
    stream.set_defaults(func=cmd_stream)

    replay = sub.add_parser("replay", help="replay a CSV of raw samples to ingest")
    replay.add_argument("--csv", default="data/sample/sample_links.csv")
    replay.add_argument("--host", default=os.environ.get("INGEST_HOST", "127.0.0.1"))
    replay.add_argument("--port", type=int, default=int(os.environ.get("INGEST_PORT", "9000")))
    replay.add_argument("--rate", type=float, default=1.0)
    replay.add_argument("--retime", action="store_true", help="shift timestamps to now")
    replay.add_argument("--loop", action="store_true")
    replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
