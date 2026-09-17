# optics-softfail

Predict optical **soft failures** — EDFA pump aging, fiber pinch, dirty connectors,
laser bias drift — and re-route traffic *before* the link goes down.

A hybrid **C++17 + Python 3.11** prototype: a C++ ingest process does the hot-path
parsing and windowing, Python does the unsupervised ML, the self-heal policy, and the
stub PCE. IPC is plain **newline-delimited JSON over TCP**.

---

## 1. Problem: soft failure vs. LOS

Optical links rarely die instantly. They *degrade*: an EDFA pump ages and its noise
figure creeps up, a patch cord gets pinched, a connector picks up dust, a
transceiver's laser bias current drifts. OSNR sags, pre-FEC BER climbs, and only at
the very end does the receiver declare **LOS (loss of signal)**.

Operations today largely reacts to that final step. By then traffic is already down
and the restoration clock is running.

This prototype watches the degradation instead:

- Ingests 1 Hz telemetry — **OSNR (dB)**, **pre-FEC BER**, **laser bias current (mA)**,
  plus **EDFA pump current (mA)** and **Rx power (dBm)**.
- Learns a **nominal baseline** with unsupervised models (Isolation Forest + a PyTorch
  autoencoder). No labelled production traces required.
- Scores each 60 s window and maps the score to **P(hard failure within 600 s)** —
  the **10-minute horizon**.
- When the policy fires, it emits a `RerouteRequest` over **REST and gRPC** to move
  traffic onto the protection lightpath, while the working path is still carrying
  traffic.

**Hard failure** (the thing we predict) is defined as `osnr_db < 8.0` **OR**
`ber > 1e-3` — i.e. LOS or uncorrectable FEC. A window is a horizon positive if any
sample in the next 600 s on the same link meets that definition.

---

## 2. Architecture

```
Python producer  --JSONL TCP :9000-->  C++ optics_ingest  --JSONL TCP :9001-->  Python infer
                                                                              |
                                                                              v
                                                                    Python healer
                                                                              |
                                              REST :8080 / gRPC :8081 -->  Python path_manager
```

| Process | Language | Role |
|---|---|---|
| `producer` | Python | GNPy-like synthetic generator or CSV replay → JSONL to ingest |
| `optics_ingest` | C++17 | Parse, validate, rolling 60 s window, emit feature frames |
| `infer` | Python (FastAPI `:8090` + TCP `:9001`) | Score windows; POST observations to the healer |
| `healer` | Python (FastAPI `:8091`) | Threshold + hysteresis, pick the alternate path, call the PCE |
| `path_manager` | Python (FastAPI `:8080` + gRPC `:8081`) | Stub PCE; applies topology; logs reroutes |

Services bind `0.0.0.0` under Docker and `127.0.0.1` on the host.

Repository layout:

```
apps/producer        synthetic + replay CLI
apps/ingest_cpp      C++17 ingest (CMake, nlohmann/json v3.11.3, CTest)
apps/infer           scoring service
apps/healer          self-heal policy
apps/path_manager    stub PCE (REST + gRPC)
ml/                  features, Isolation Forest, autoencoder, horizon head, training
data/generator       closed-form GNPy-like optical toy
data/sample          committed sample CSV (3 links x 20 min @ 1 Hz)
configs/             thresholds.yaml, topology.yaml
proto/optics/v1      telemetry.proto, path_control.proto
models/              committed trained artifacts
tests/               pytest suite
```

---

## 3. Quick start (clone to run)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make proto        # gRPC stubs into generated/
make ingest       # cmake build of the C++ ingest
make data         # synthetic corpus + refreshed sample CSV
make train        # writes models/*
make test         # ctest + pytest
```

`make ingest` needs `cmake` and a C++17 compiler; `pip install cmake` works if your
system package manager does not have one. The build pulls `nlohmann/json` v3.11.3 via
CMake `FetchContent` (or uses an installed copy if one is found).

`make venv` does the first two lines for you. If the `torch` wheel does not resolve on
your platform, install the CPU build explicitly first:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Individual services (host mode):

```bash
make run-path-manager   # :8080 REST, :8081 gRPC
make run-healer         # :8091
make run-infer          # :9001 frames, :8090 API
./apps/ingest_cpp/build/optics_ingest --listen 127.0.0.1:9000 --downstream 127.0.0.1:9001
make run-producer       # streams the compressed demo scenario
```

---

## 4. Docker

```bash
docker compose up --build
```

Published ports: `ingest 9000`, `infer 9001 + 8090`, `healer 8091`,
`path_manager 8080 + 8081`. Startup is ordered by healthchecks
(`path_manager → healer → infer → ingest → producer`), and the producer runs with
`OPTICS_DEMO=1`.

---

## 5. `make demo`

```bash
make demo         # docker compose, wait for a reroute, print it, tear down
make demo-local   # same flow without Docker (background PIDs)
```

`scripts/demo.sh` greps the service logs for `REROUTE_EMITTED` / `"accepted": true`
with a 120 s timeout and exits non-zero on timeout.

**Demo schedule (`OPTICS_DEMO=1`).** `L1` is the working path and starts EDFA aging at
`t = 20 s` with `tau = 45 s`; `L2` is protection and `L3` a second working path, both
nominal. Hard failure lands on `L1` at ≈77 s — after the first 60 s window exists, so
the model sees a genuine ~18 s soft-failure window, fires, and traffic moves to
`P_prot_L2` while `L1` is still up.

Training uses realistic timescales instead: aging onsets at 30 min with `tau` 20–40 min
so hard failure arrives 12–25 min later and 10-minute horizon labels actually exist.
`OPTICS_DEMO=1` (or `--demo`) selects the compressed schedule; the generator's
`training_specs()` is the slow one.

---

## 6. Telemetry fields

Raw sample (producer → ingest, one JSON object per line):

```json
{"ts_unix_ms": 1710000000000, "link_id": "L1", "channel_id": "C1",
 "osnr_db": 18.4, "ber": 1.2e-6, "laser_bias_ma": 42.1,
 "edfa_pump_ma": 180.0, "rx_power_dbm": -12.3, "label": "nominal"}
```

| Field | Type | Required | Range / notes |
|---|---|---|---|
| `ts_unix_ms` | int | yes | monotonic per `(link_id, channel_id)` |
| `link_id` | string | yes | must match `configs/topology.yaml` |
| `channel_id` | string | no | defaults to `"C1"` |
| `osnr_db` | float | yes | typically 8–25 dB; LOS ≈ 0 |
| `ber` | float | yes | `> 0`; features use `log10(max(ber, 1e-15))` |
| `laser_bias_ma` | float | yes | nominal 40 mA |
| `edfa_pump_ma` | float | no | `null` → ingest fills `0` |
| `rx_power_dbm` | float | no | `null` → ingest fills `-99` |
| `label` | string | train/eval only | `nominal`, `edfa_aging`, `fiber_pinch`, `dirty_connector`, `laser_drift`, `los`; stripped before live inference |

Feature frame (ingest → infer) carries the 60-sample `seq` for five channels
(`osnr_db`, `log10_ber`, `laser_bias_ma`, `edfa_pump_ma`, `rx_power_dbm`) plus eight
`stats`: `osnr_mean/std/slope`, `log10_ber_mean/std/slope`, `laser_bias_mean/slope`.
Slopes are OLS against the time index `0..59`; incomplete windows are dropped.

Samples are rejected by the C++ validator when a required field is missing, a number is
non-finite, or `ber <= 0`.

---

## 7. Models

All three artifacts are frozen at training time and committed under `models/`.

1. **Isolation Forest** (`ml/models/isolation_forest.py`) over the 8 window stats —
   `n_estimators=200`, `contamination=0.02`, `random_state=42`, fitted on **nominal
   windows only**. `if_score = -decision_function(x)`, min-max calibrated against a
   nominal holdout to `s_if ∈ [0, 1]`.
2. **Autoencoder** (`ml/models/autoencoder.py`) — PyTorch MLP `300 → 64 → 16 → 64 → 300`
   over the flattened `[60, 5]` window, z-scored with the train-set scaler
   (`models/scaler.joblib`). MSE loss, Adam `1e-3`, batch 64, ≤20 epochs, CPU only.
   `s_ae` = reconstruction error over the 98th percentile of the nominal holdout error,
   clipped to `[0, 1]`.
3. **Horizon head** (`ml/models/horizon.py`):

   ```
   s = 0.5 * s_if + 0.5 * s_ae
   p_fail_10m = 1 / (1 + exp(-k * (s - s0)))
   ```

   `k` and `s0` come from a logistic regression on synthetic labelled windows
   (label = hard failure within the next 600 s), then `s0` is nudged so the
   **calibration gate** holds: on nominal-only holdout traffic, **FPR ≤ 0.02** at the
   operating threshold `p_fail_10m >= 0.7`. That logistic fit is the only supervised
   component, it only ever sees synthetic labels, and the live path runs frozen weights.

Artifacts: `models/iforest.joblib`, `models/autoencoder.pt` +
`models/autoencoder_meta.json`, `models/scaler.joblib`, `models/horizon.json`, plus a
`models/train_report.json` with the calibration numbers.

**Self-heal policy** (`configs/thresholds.yaml`): fire after **3 consecutive** windows
with `p_fail_10m >= 0.7` on the same link, then hold off for **300 s** (hysteresis) so
the system cannot flap. The healer resolves `from_path` (the working path containing the
link) and `to_path` (`protection[link_id]`), calls the PCE over **both** REST and gRPC,
and logs one `REROUTE_EMITTED` line. The path manager rejects the reroute if the
protection path has no spare capacity.

---

## 8. Non-goals

Deliberately **not** in this prototype:

- Real DWDM hardware, GNPy as a live subprocess, LUX-Optical downloads.
- **Nokia WaveSuite** or any other production NMS/OSS, PCE/BGP-LS, OpenConfig/gNMI
  southbound. `path_manager` is a stub PCE with in-memory state.
- Supervised classifiers that need labelled production traces; labels exist only inside
  the synthetic generator, for evaluation.
- Multi-vendor inventory, GUI dashboards (stdout + JSON logs), Kafka, Redis,
  Kubernetes, ZeroMQ.

The optical model in `data/generator/gnpy_like.py` is a closed-form toy inspired by
Gaussian-noise-model intuition — it does not call GNPy and is not a substitute for it.

---

## 9. License

MIT — see [LICENSE](LICENSE).
