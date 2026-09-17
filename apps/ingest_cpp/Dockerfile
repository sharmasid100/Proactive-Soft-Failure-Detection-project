# C++17 ingest: build with cmake + FetchContent(nlohmann/json v3.11.3), ship the binary.
FROM gcc:13-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . ./
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j "$(nproc)" \
    && ctest --test-dir build --output-on-failure

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/optics_ingest /usr/local/bin/optics_ingest
ENTRYPOINT ["/usr/local/bin/optics_ingest"]
CMD ["--listen", "0.0.0.0:9000", "--downstream", "infer:9001"]
