# EvalHub Adapters Makefile
# Build container images for various evaluation framework adapters

# Variables
REGISTRY ?= quay.io/eval-hub
BUILD_TOOL ?= podman
VERSION ?= latest
PYTHON_VERSION ?= 3.12

# Image names
IMAGE_DATABENCH = $(REGISTRY)/community-databench:$(VERSION)
IMAGE_FOLLOWBENCH = $(REGISTRY)/community-followbench:$(VERSION)
IMAGE_LIGHTEVAL = $(REGISTRY)/community-lighteval:$(VERSION)
IMAGE_GUIDELLM = $(REGISTRY)/community-guidellm:$(VERSION)
IMAGE_MTEB = $(REGISTRY)/community-mteb:$(VERSION)
IMAGE_INSPECT = $(REGISTRY)/community-inspect:$(VERSION)
IMAGE_DEEPEVAL = $(REGISTRY)/community-deepeval:$(VERSION)
IMAGE_RAGAS = $(REGISTRY)/community-ragas:$(VERSION)
IMAGE_SWEBENCH = $(REGISTRY)/community-swebench:$(VERSION)
IMAGE_RULER = $(REGISTRY)/community-ruler:$(VERSION)
IMAGE_NEMO_GUARDRAILS = $(REGISTRY)/community-nemo-guardrails:$(VERSION)
IMAGE_TOOLEVAL = $(REGISTRY)/community-tooleval:$(VERSION)
# Fixture image tag matches tool-server/deploy.yaml; full-cache appends -full.
# Use evalhub (not REGISTRY's eval-hub) so Makefile output matches deploy.yaml / CI.
TOOLBENCH_SERVER_REGISTRY ?= quay.io/evalhub
TOOLBENCH_SERVER_VERSION ?= 0.3.0
IMAGE_TOOLBENCH_SERVER = $(TOOLBENCH_SERVER_REGISTRY)/community-toolbench-server:$(TOOLBENCH_SERVER_VERSION)
IMAGE_PROMPTFOO = $(REGISTRY)/community-promptfoo:$(VERSION)

# Default target
.PHONY: help
help:
	@echo "EvalHub Adapters Build Targets"
	@echo "==============================="
	@echo ""
	@echo "Image Build:"
	@echo "  make image-lighteval    - Build LightEval adapter image"
	@echo "  make image-guidellm     - Build GuideLLM adapter image"
	@echo "  make image-mteb         - Build MTEB adapter image"
	@echo "  make image-inspect      - Build Inspect AI adapter image"
	@echo "  make image-deepeval     - Build DeepEval adapter image"
	@echo "  make image-ragas        - Build RAGAS adapter image"
	@echo "  make image-tooleval     - Build ToolEval adapter image"
	@echo "  make image-toolbench-server - Build ToolBench tool server ($(TOOLBENCH_SERVER_VERSION); FETCH_FULL_CACHE=1 -> $(TOOLBENCH_SERVER_VERSION)-full)"
	@echo "  make image-swebench     - Build SWE-bench adapter image"
	@echo "  make image-ruler        - Build RULER adapter image"
	@echo "  make images             - Build all adapter images"
	@echo ""
	@echo "Image Push:"
	@echo "  make push-lighteval     - Push LightEval adapter image"
	@echo "  make push-guidellm      - Push GuideLLM adapter image"
	@echo "  make push-mteb          - Push MTEB adapter image"
	@echo "  make push-inspect       - Push Inspect AI adapter image"
	@echo "  make push-deepeval      - Push DeepEval adapter image"
	@echo "  make push-ragas         - Push RAGAS adapter image"
	@echo "  make push-tooleval      - Push ToolEval adapter image"
	@echo "  make push-toolbench-server - Push ToolBench tool server image"
	@echo "  make push-swebench      - Push SWE-bench adapter image"
	@echo "  make push-ruler         - Push RULER adapter image"
	@echo "  make push-images        - Push all adapter images"
	@echo ""
	@echo "Clean:"
	@echo "  make clean-lighteval    - Remove LightEval adapter image"
	@echo "  make clean-guidellm     - Remove GuideLLM adapter image"
	@echo "  make clean-mteb         - Remove MTEB adapter image"
	@echo "  make clean-inspect      - Remove Inspect AI adapter image"
	@echo "  make clean-deepeval     - Remove DeepEval adapter image"
	@echo "  make clean-ragas        - Remove RAGAS adapter image"
	@echo "  make clean-tooleval     - Remove ToolEval adapter image"
	@echo "  make clean-toolbench-server - Remove ToolBench tool server image"
	@echo "  make clean-swebench     - Remove SWE-bench adapter image"
	@echo "  make clean-ruler        - Remove RULER adapter image"
	@echo "  make clean-images       - Remove all adapter images"
	@echo ""
	@echo "Test:"
	@echo "  make test-guidellm     - Run GuideLLM adapter tests"
	@echo "  make test-lighteval    - Run LightEval adapter tests"
	@echo "  make test-mteb         - Run MTEB adapter tests"
	@echo "  make test-clear        - Run CLEAR adapter tests"
	@echo "  make test-inspect      - Run Inspect AI adapter tests"
	@echo "  make test-deepeval     - Run DeepEval adapter tests"
	@echo "  make test-ragas        - Run RAGAS adapter tests"
	@echo "  make test-tooleval     - Run ToolEval adapter tests"
	@echo "  make test-ruler        - Run RULER adapter tests"
	@echo "  make tests             - Run all adapter tests"
	@echo ""
	@echo "Variables:"
	@echo "  REGISTRY=$(REGISTRY)"
	@echo "  BUILD_TOOL=$(BUILD_TOOL)"
	@echo "  VERSION=$(VERSION)"
	@echo "  PYTHON_VERSION=$(PYTHON_VERSION)"
	@echo ""
	@echo "Example:"
	@echo "  make image-lighteval REGISTRY=localhost:5000 VERSION=dev"

# Build targets
.PHONY: image-followbench
image-followbench:
	@echo "Building FollowBench adapter image..."
	cd adapters/followbench && \
	$(BUILD_TOOL) build -t $(IMAGE_FOLLOWBENCH) -f Containerfile .
	@echo "✅ Built: $(IMAGE_FOLLOWBENCH)"

.PHONY: image-lighteval
image-lighteval:
	@echo "Building LightEval adapter image..."
	cd adapters/lighteval && \
	$(BUILD_TOOL) build -t $(IMAGE_LIGHTEVAL) -f Containerfile .
	@echo "✅ Built: $(IMAGE_LIGHTEVAL)"

.PHONY: image-guidellm
image-guidellm:
	@echo "Building GuideLLM adapter image..."
	cd adapters/guidellm && \
	$(BUILD_TOOL) build -t $(IMAGE_GUIDELLM) -f Containerfile .
	@echo "✅ Built: $(IMAGE_GUIDELLM)"

.PHONY: image-mteb
image-mteb:
	@echo "Building MTEB adapter image..."
	cd adapters/mteb && \
	$(BUILD_TOOL) build -t $(IMAGE_MTEB) -f Containerfile .
	@echo "✅ Built: $(IMAGE_MTEB)"

.PHONY: image-inspect
image-inspect:
	@echo "Building Inspect AI adapter image..."
	cd adapters/inspect && \
	$(BUILD_TOOL) build -t $(IMAGE_INSPECT) -f Containerfile .
	@echo "✅ Built: $(IMAGE_INSPECT)"

.PHONY: images
images: image-lighteval image-guidellm image-mteb image-inspect
.PHONY: image-deepeval
image-deepeval:
	@echo "Building DeepEval adapter image..."
	cd adapters/deepeval && \
	$(BUILD_TOOL) build -t $(IMAGE_DEEPEVAL) -f Containerfile .
	@echo "✅ Built: $(IMAGE_DEEPEVAL)"

.PHONY: images
images: image-lighteval image-guidellm image-mteb image-deepeval
.PHONY: image-ragas
image-ragas:
	@echo "Building RAGAS adapter image..."
	cd adapters/ragas && \
	$(BUILD_TOOL) build -t $(IMAGE_RAGAS) -f Containerfile .
	@echo "✅ Built: $(IMAGE_RAGAS)"

.PHONY: image-swebench
image-swebench:
	@echo "Building SWE-bench adapter image..."
	cd adapters/swebench && \
	$(BUILD_TOOL) build -t $(IMAGE_SWEBENCH) -f Containerfile .
	@echo "✅ Built: $(IMAGE_SWEBENCH)"

.PHONY: images
images: image-lighteval image-guidellm image-mteb image-ragas image-swebench image-ruler image-nemo-guardrails image-promptfoo image-tooleval image-toolbench-server
	@echo "✅ All adapter images built"

# Push targets
.PHONY: push-lighteval
push-lighteval:
	@echo "Pushing LightEval adapter image..."
	$(BUILD_TOOL) push $(IMAGE_LIGHTEVAL)
	@echo "✅ Pushed: $(IMAGE_LIGHTEVAL)"

.PHONY: push-guidellm
push-guidellm:
	@echo "Pushing GuideLLM adapter image..."
	$(BUILD_TOOL) push $(IMAGE_GUIDELLM)
	@echo "✅ Pushed: $(IMAGE_GUIDELLM)"

.PHONY: push-mteb
push-mteb:
	@echo "Pushing MTEB adapter image..."
	$(BUILD_TOOL) push $(IMAGE_MTEB)
	@echo "✅ Pushed: $(IMAGE_MTEB)"

.PHONY: push-inspect
push-inspect:
	@echo "Pushing Inspect AI adapter image..."
	$(BUILD_TOOL) push $(IMAGE_INSPECT)
	@echo "✅ Pushed: $(IMAGE_INSPECT)"

.PHONY: push-images
push-images: push-lighteval push-guidellm push-mteb push-inspect
.PHONY: push-deepeval
push-deepeval:
	@echo "Pushing DeepEval adapter image..."
	$(BUILD_TOOL) push $(IMAGE_DEEPEVAL)
	@echo "✅ Pushed: $(IMAGE_DEEPEVAL)"

.PHONY: push-images
push-images: push-lighteval push-guidellm push-mteb push-deepeval
.PHONY: push-ragas
push-ragas:
	@echo "Pushing RAGAS adapter image..."
	$(BUILD_TOOL) push $(IMAGE_RAGAS)
	@echo "✅ Pushed: $(IMAGE_RAGAS)"

.PHONY: push-swebench
push-swebench:
	@echo "Pushing SWE-bench adapter image..."
	$(BUILD_TOOL) push $(IMAGE_SWEBENCH)
	@echo "✅ Pushed: $(IMAGE_SWEBENCH)"

.PHONY: push-images
push-images: push-lighteval push-guidellm push-mteb push-ragas push-swebench push-ruler push-nemo-guardrails push-promptfoo push-tooleval push-toolbench-server
	@echo "✅ All adapter images pushed"

# Clean targets
.PHONY: clean-lighteval
clean-lighteval:
	@echo "Removing LightEval adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_LIGHTEVAL) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_LIGHTEVAL)"

.PHONY: clean-guidellm
clean-guidellm:
	@echo "Removing GuideLLM adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_GUIDELLM) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_GUIDELLM)"

.PHONY: clean-mteb
clean-mteb:
	@echo "Removing MTEB adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_MTEB) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_MTEB)"

.PHONY: clean-inspect
clean-inspect:
	@echo "Removing Inspect AI adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_INSPECT) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_INSPECT)"

.PHONY: clean-images
clean-images: clean-lighteval clean-guidellm clean-mteb clean-inspect
.PHONY: clean-deepeval
clean-deepeval:
	@echo "Removing DeepEval adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_DEEPEVAL) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_DEEPEVAL)"

.PHONY: clean-images
clean-images: clean-lighteval clean-guidellm clean-mteb clean-deepeval
.PHONY: clean-ragas
clean-ragas:
	@echo "Removing RAGAS adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_RAGAS) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_RAGAS)"

.PHONY: clean-swebench
clean-swebench:
	@echo "Removing SWE-bench adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_SWEBENCH) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_SWEBENCH)"

.PHONY: clean-images
clean-images: clean-lighteval clean-guidellm clean-mteb clean-ragas clean-swebench clean-ruler clean-nemo-guardrails clean-promptfoo clean-tooleval clean-toolbench-server
	@echo "✅ All adapter images removed"

# Development targets
.PHONY: build-and-push-lighteval
build-and-push-lighteval: image-lighteval push-lighteval
	@echo "✅ LightEval adapter built and pushed"

.PHONY: build-and-push-guidellm
build-and-push-guidellm: image-guidellm push-guidellm
	@echo "✅ GuideLLM adapter built and pushed"

.PHONY: build-and-push-mteb
build-and-push-mteb: image-mteb push-mteb
	@echo "✅ MTEB adapter built and pushed"

.PHONY: build-and-push-deepeval
build-and-push-deepeval: image-deepeval push-deepeval
	@echo "✅ DeepEval adapter built and pushed"
.PHONY: build-and-push-swebench
build-and-push-swebench: image-swebench push-swebench
	@echo "✅ SWE-bench adapter built and pushed"

.PHONY: build-and-push-all
build-and-push-all: images push-images
	@echo "✅ All adapters built and pushed"

# Test targets
.PHONY: test-guidellm
test-guidellm:
	@echo "Running GuideLLM adapter tests..."
	cd adapters/guidellm && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ GuideLLM tests passed"

.PHONY: test-lighteval
test-lighteval:
	@echo "Running LightEval adapter tests..."
	cd adapters/lighteval && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ LightEval tests passed"

.PHONY: test-mteb
test-mteb:
	@echo "Running MTEB adapter tests..."
	cd adapters/mteb && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ MTEB tests passed"

.PHONY: test-clear
test-clear:
	@echo "Running CLEAR adapter tests..."
	cd adapters/clear && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ CLEAR tests passed"

.PHONY: test-inspect
test-inspect:
	@echo "Running Inspect AI adapter tests..."
	cd adapters/inspect && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ Inspect AI tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-inspect
.PHONY: test-deepeval
test-deepeval:
	@echo "Running DeepEval adapter tests..."
	cd adapters/deepeval && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ DeepEval tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-deepeval
.PHONY: test-ragas
test-ragas:
	@echo "Running RAGAS adapter tests..."
	cd adapters/ragas && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ RAGAS tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-ragas test-ruler test-nemo-guardrails test-promptfoo test-swebench test-tooleval
	@echo "✅ All adapter tests passed"
.PHONY: test-swebench
test-swebench:
	@echo "Running SWE-bench adapter tests..."
	cd adapters/swebench && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ SWE-bench tests passed"

.PHONY: image-ruler
image-ruler:
	@echo "Building RULER adapter image..."
	cd adapters/ruler && \
	$(BUILD_TOOL) build -t $(IMAGE_RULER) -f Containerfile .
	@echo "✅ Built: $(IMAGE_RULER)"

.PHONY: push-ruler
push-ruler:
	@echo "Pushing RULER adapter image..."
	$(BUILD_TOOL) push $(IMAGE_RULER)
	@echo "✅ Pushed: $(IMAGE_RULER)"

.PHONY: clean-ruler
clean-ruler:
	@echo "Removing RULER adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_RULER) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_RULER)"

.PHONY: build-and-push-ruler
build-and-push-ruler: image-ruler push-ruler
	@echo "✅ RULER adapter built and pushed"

.PHONY: test-ruler
test-ruler:
	@echo "Running RULER adapter tests..."
	cd adapters/ruler && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ RULER tests passed"

.PHONY: image-nemo-guardrails
image-nemo-guardrails:
	@echo "Building NeMo Guardrails adapter image..."
	cd adapters/nemo-guardrails && \
	$(BUILD_TOOL) build -t $(IMAGE_NEMO_GUARDRAILS) -f Containerfile .
	@echo "✅ Built: $(IMAGE_NEMO_GUARDRAILS)"

.PHONY: push-nemo-guardrails
push-nemo-guardrails:
	@echo "Pushing NeMo Guardrails adapter image..."
	$(BUILD_TOOL) push $(IMAGE_NEMO_GUARDRAILS)
	@echo "✅ Pushed: $(IMAGE_NEMO_GUARDRAILS)"

.PHONY: clean-nemo-guardrails
clean-nemo-guardrails:
	@echo "Removing NeMo Guardrails adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_NEMO_GUARDRAILS) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_NEMO_GUARDRAILS)"

.PHONY: build-and-push-nemo-guardrails
build-and-push-nemo-guardrails: image-nemo-guardrails
	$(MAKE) push-nemo-guardrails
	@echo "✅ NeMo Guardrails adapter built and pushed"

.PHONY: test-nemo-guardrails
test-nemo-guardrails:
	@echo "Running NeMo Guardrails adapter tests..."
	cd adapters/nemo-guardrails && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ NeMo Guardrails tests passed"

.PHONY: image-promptfoo
image-promptfoo:
	@echo "Building promptfoo adapter image..."
	cd adapters/promptfoo && \
	$(BUILD_TOOL) build -t $(IMAGE_PROMPTFOO) -f Containerfile .
	@echo "✅ Built: $(IMAGE_PROMPTFOO)"

.PHONY: push-promptfoo
push-promptfoo:
	@echo "Pushing promptfoo adapter image..."
	$(BUILD_TOOL) push $(IMAGE_PROMPTFOO)
	@echo "✅ Pushed: $(IMAGE_PROMPTFOO)"

.PHONY: clean-promptfoo
clean-promptfoo:
	@echo "Removing promptfoo adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_PROMPTFOO) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_PROMPTFOO)"

.PHONY: build-and-push-promptfoo
build-and-push-promptfoo: image-promptfoo push-promptfoo
	@echo "✅ promptfoo adapter built and pushed"

.PHONY: test-promptfoo
test-promptfoo:
	@echo "Running promptfoo adapter tests..."
	cd adapters/promptfoo && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ promptfoo tests passed"

.PHONY: image-tooleval
image-tooleval:
	@echo "Building ToolEval adapter image..."
	cd adapters/tooleval && \
	$(BUILD_TOOL) build -t $(IMAGE_TOOLEVAL) -f Containerfile .
	@echo "✅ Built: $(IMAGE_TOOLEVAL)"

.PHONY: push-tooleval
push-tooleval:
	@echo "Pushing ToolEval adapter image..."
	$(BUILD_TOOL) push $(IMAGE_TOOLEVAL)
	@echo "✅ Pushed: $(IMAGE_TOOLEVAL)"

.PHONY: clean-tooleval
clean-tooleval:
	@echo "Removing ToolEval adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_TOOLEVAL) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_TOOLEVAL)"

# Optional: FETCH_FULL_CACHE=1 bakes the HF StableToolBench tools+cache into the image.
# Fixture and full-cache images use distinct tags so IfNotPresent cannot mix them.
FETCH_FULL_CACHE ?= 0
ifeq ($(FETCH_FULL_CACHE),1)
IMAGE_TOOLBENCH_SERVER = $(TOOLBENCH_SERVER_REGISTRY)/community-toolbench-server:$(TOOLBENCH_SERVER_VERSION)-full
endif

.PHONY: image-toolbench-server
image-toolbench-server:
	@echo "Building ToolBench tool server image (FETCH_FULL_CACHE=$(FETCH_FULL_CACHE)) -> $(IMAGE_TOOLBENCH_SERVER)"
	cd adapters/tooleval/tool-server && \
	$(BUILD_TOOL) build -t $(IMAGE_TOOLBENCH_SERVER) -f Containerfile \
		--build-arg FETCH_FULL_CACHE=$(FETCH_FULL_CACHE) .
	@echo "✅ Built: $(IMAGE_TOOLBENCH_SERVER)"

.PHONY: push-toolbench-server
push-toolbench-server:
	@echo "Pushing ToolBench tool server image..."
	$(BUILD_TOOL) push $(IMAGE_TOOLBENCH_SERVER)
	@echo "✅ Pushed: $(IMAGE_TOOLBENCH_SERVER)"

.PHONY: clean-toolbench-server
clean-toolbench-server:
	@echo "Removing ToolBench tool server image..."
	$(BUILD_TOOL) rmi $(IMAGE_TOOLBENCH_SERVER) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_TOOLBENCH_SERVER)"

.PHONY: test-tooleval
test-tooleval:
	@echo "Running ToolEval adapter tests..."
	cd adapters/tooleval && \
	test -d .venv || uv venv --python $(PYTHON_VERSION) .venv && \
	uv pip install --quiet --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ ToolEval tests passed"

# DataBench adapter
.PHONY: image-databench push-databench clean-databench test-databench
image-databench:
	$(BUILD_TOOL) build -t $(IMAGE_DATABENCH) -f adapters/databench/Containerfile adapters/databench
push-databench:
	$(BUILD_TOOL) push $(IMAGE_DATABENCH)
clean-databench:
	$(BUILD_TOOL) rmi $(IMAGE_DATABENCH)
test-databench:
	cd adapters/databench && \
	(test -d .venv || uv venv --python $(PYTHON_VERSION) .venv) && \
	uv pip install --python .venv/bin/python -r requirements.txt -r requirements-test.txt && \
	.venv/bin/python -m pytest tests/ -v
images: image-databench
push-images: push-databench
clean-images: clean-databench
tests: test-databench
