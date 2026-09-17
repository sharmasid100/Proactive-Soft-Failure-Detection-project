# Shared image for the Python services (producer, infer, healer, path_manager).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# CPU-only torch wheel keeps the image small; the rest comes from PyPI.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# gRPC stubs are gitignored, so generate them at build time.
RUN bash scripts/gen_protos.sh

CMD ["python", "-m", "apps.path_manager.main"]
