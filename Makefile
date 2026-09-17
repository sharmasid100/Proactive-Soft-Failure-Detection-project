# optics-softfail — reproducible build / train / test / demo
.DEFAULT_GOAL := help
SHELL := /bin/bash

ROOT := $(shell pwd)
export PYTHONPATH := $(ROOT)

VENV := $(ROOT)/.venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(PYTHON) -m pip

INGEST_DIR := apps/ingest_cpp
INGEST_BUILD := $(INGEST_DIR)/build
DATA_DIR := data/synthetic
MODELS_DIR := models
HOURS ?= 4

.PHONY: help venv install proto data train ingest test test-py test-cpp demo demo-local run-% clean distclean

help: ## show available targets
	@grep -hE '^[a-zA-Z_%-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## create .venv with pinned dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

install: ## install pinned dependencies into the active interpreter
	$(PIP) install -r requirements.txt

proto: ## generate Python gRPC stubs into generated/
	PYTHON=$(PYTHON) bash scripts/gen_protos.sh

data: ## generate the synthetic corpus and refresh the sample CSV
	$(PYTHON) -m apps.producer.main generate --out $(DATA_DIR) --hours $(HOURS)

train: ## fit and persist models/ (falls back to the sample CSV)
	$(PYTHON) -m ml.train.train --data $(DATA_DIR) --out $(MODELS_DIR)

ingest: ## configure and build the C++ ingest binary
	cmake -S $(INGEST_DIR) -B $(INGEST_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(INGEST_BUILD) -j

test: test-cpp test-py ## run CTest and pytest

test-cpp: ## build and run the C++ unit tests
	cmake -S $(INGEST_DIR) -B $(INGEST_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(INGEST_BUILD) -j --target test_window
	ctest --test-dir $(INGEST_BUILD) --output-on-failure

test-py: ## run pytest
	$(PYTHON) -m pytest -q

demo: ## docker compose up, wait for a reroute, print it, tear down
	bash scripts/demo.sh

demo-local: ## same demo without Docker (background PIDs)
	bash scripts/demo.sh --local

run-producer: ## stream the compressed demo scenario to ingest
	$(PYTHON) -m apps.producer.main stream --host 127.0.0.1 --port 9000 --demo

run-infer: ## run the infer service (:9001 frames, :8090 API)
	$(PYTHON) -m apps.infer.main --host 127.0.0.1

run-healer: ## run the healer (:8091)
	HEALER_HOST=127.0.0.1 $(PYTHON) -m apps.healer.main

run-path-manager: ## run the path manager (:8080 REST, :8081 gRPC)
	PATH_MANAGER_HOST=127.0.0.1 $(PYTHON) -m apps.path_manager.main

clean: ## remove build output, generated stubs and caches
	rm -rf $(INGEST_BUILD) generated .pytest_cache .cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean ## also remove the venv and the generated corpus
	rm -rf $(VENV) $(DATA_DIR)/*.parquet $(DATA_DIR)/*.jsonl $(DATA_DIR)/*.csv
