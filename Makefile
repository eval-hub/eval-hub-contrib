# EvalHub Adapters Makefile
# Build container images for various evaluation framework adapters

# Variables
REGISTRY ?= quay.io/eval-hub
BUILD_TOOL ?= podman
VERSION ?= latest

# Image names
IMAGE_LIGHTEVAL = $(REGISTRY)/community-lighteval:$(VERSION)
IMAGE_GUIDELLM = $(REGISTRY)/community-guidellm:$(VERSION)
IMAGE_MTEB = $(REGISTRY)/community-mteb:$(VERSION)
IMAGE_INSPECT = $(REGISTRY)/community-inspect:$(VERSION)
IMAGE_DEEPEVAL = $(REGISTRY)/community-deepeval:$(VERSION)
IMAGE_RAGAS = $(REGISTRY)/community-ragas:$(VERSION)
IMAGE_SWEBENCH = $(REGISTRY)/community-swebench:$(VERSION)

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
	@echo "  make image-swebench     - Build SWE-bench adapter image"
	@echo "  make images             - Build all adapter images"
	@echo ""
	@echo "Image Push:"
	@echo "  make push-lighteval     - Push LightEval adapter image"
	@echo "  make push-guidellm      - Push GuideLLM adapter image"
	@echo "  make push-mteb          - Push MTEB adapter image"
	@echo "  make push-inspect       - Push Inspect AI adapter image"
	@echo "  make push-deepeval      - Push DeepEval adapter image"
	@echo "  make push-ragas         - Push RAGAS adapter image"
	@echo "  make push-swebench      - Push SWE-bench adapter image"
	@echo "  make push-images        - Push all adapter images"
	@echo ""
	@echo "Clean:"
	@echo "  make clean-lighteval    - Remove LightEval adapter image"
	@echo "  make clean-guidellm     - Remove GuideLLM adapter image"
	@echo "  make clean-mteb         - Remove MTEB adapter image"
	@echo "  make clean-inspect      - Remove Inspect AI adapter image"
	@echo "  make clean-deepeval     - Remove DeepEval adapter image"
	@echo "  make clean-ragas        - Remove RAGAS adapter image"
	@echo "  make clean-swebench     - Remove SWE-bench adapter image"
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
	@echo "  make tests             - Run all adapter tests"
	@echo ""
	@echo "Variables:"
	@echo "  REGISTRY=$(REGISTRY)"
	@echo "  BUILD_TOOL=$(BUILD_TOOL)"
	@echo "  VERSION=$(VERSION)"
	@echo ""
	@echo "Example:"
	@echo "  make image-lighteval REGISTRY=localhost:5000 VERSION=dev"

# Build targets
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
images: image-lighteval image-guidellm image-mteb image-ragas image-swebench
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
push-images: push-lighteval push-guidellm push-mteb push-ragas push-swebench
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
clean-images: clean-lighteval clean-guidellm clean-mteb clean-ragas clean-swebench
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
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ GuideLLM tests passed"

.PHONY: test-lighteval
test-lighteval:
	@echo "Running LightEval adapter tests..."
	cd adapters/lighteval && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ LightEval tests passed"

.PHONY: test-mteb
test-mteb:
	@echo "Running MTEB adapter tests..."
	cd adapters/mteb && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ MTEB tests passed"

.PHONY: test-clear
test-clear:
	@echo "Running CLEAR adapter tests..."
	cd adapters/clear && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ CLEAR tests passed"

.PHONY: test-inspect
test-inspect:
	@echo "Running Inspect AI adapter tests..."
	cd adapters/inspect && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ Inspect AI tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-inspect
.PHONY: test-deepeval
test-deepeval:
	@echo "Running DeepEval adapter tests..."
	cd adapters/deepeval && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ DeepEval tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-deepeval
.PHONY: test-ragas
test-ragas:
	@echo "Running RAGAS adapter tests..."
	cd adapters/ragas && \
	test -d .venv || python3 -m venv .venv && \
	.venv/bin/pip install --quiet -r requirements.txt -r requirements-test.txt && \
	PATH="$$(pwd)/.venv/bin:$$PATH" .venv/bin/pytest tests/ -v
	@echo "✅ RAGAS tests passed"

.PHONY: tests
tests: test-guidellm test-lighteval test-mteb test-clear test-ragas
	@echo "✅ All adapter tests passed"
.PHONY: test-swebench
test-swebench:
	@echo "Testing SWE-bench adapter..."
	cd adapters/swebench && \
	python -m venv .venv 2>/dev/null || true && \
	. .venv/bin/activate && \
	pip install -q -r requirements.txt -r requirements-test.txt && \
	pytest tests/ -v
	@echo "✅ SWE-bench adapter tests passed"
