#!/usr/bin/env bash
# Generate Python gRPC/protobuf stubs into generated/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT="generated"

mkdir -p "$OUT"
touch "$OUT/__init__.py"

"$PYTHON" -m grpc_tools.protoc \
  -I proto \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  proto/optics/v1/telemetry.proto \
  proto/optics/v1/path_control.proto

# protoc emits packages without __init__.py; add them so `generated.optics.v1...` imports.
find "$OUT" -type d -exec touch {}/__init__.py \;

# grpc_python_out emits absolute imports (`from optics.v1 import ...`); rewrite them
# to be relative to the generated/ package so `generated` alone is importable.
for f in "$OUT"/optics/v1/*_pb2_grpc.py; do
  [ -f "$f" ] || continue
  "$PYTHON" - "$f" <<'PY'
import re, sys
path = sys.argv[1]
with open(path) as fh:
    src = fh.read()
src = re.sub(r'^from optics\.v1 import', 'from generated.optics.v1 import', src, flags=re.M)
with open(path, 'w') as fh:
    fh.write(src)
PY
done

echo "protos generated into $OUT/"
