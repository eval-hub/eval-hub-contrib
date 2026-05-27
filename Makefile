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
IMAGE_DEEPEVAL = $(REGISTRY)/community-deepeval:$(VERSION)

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
	@echo "  make image-deepeval     - Build DeepEval adapter image"
	@echo "  make images             - Build all adapter images"
	@echo ""
	@echo "Image Push:"
	@echo "  make push-lighteval     - Push LightEval adapter image"
	@echo "  make push-guidellm      - Push GuideLLM adapter image"
	@echo "  make push-mteb          - Push MTEB adapter image"
	@echo "  make push-deepeval      - Push DeepEval adapter image"
	@echo "  make push-images        - Push all adapter images"
	@echo ""
	@echo "Clean:"
	@echo "  make clean-lighteval    - Remove LightEval adapter image"
	@echo "  make clean-guidellm     - Remove GuideLLM adapter image"
	@echo "  make clean-mteb         - Remove MTEB adapter image"
	@echo "  make clean-deepeval     - Remove DeepEval adapter image"
	@echo "  make clean-images       - Remove all adapter images"
	@echo ""
	@echo "Test:"
	@echo "  make test-guidellm     - Run GuideLLM adapter tests"
	@echo "  make test-lighteval    - Run LightEval adapter tests"
	@echo "  make test-mteb         - Run MTEB adapter tests"
	@echo "  make test-clear        - Run CLEAR adapter tests"
	@echo "  make test-deepeval     - Run DeepEval adapter tests"
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

.PHONY: image-deepeval
image-deepeval:
	@echo "Building DeepEval adapter image..."
	cd adapters/deepeval && \
	$(BUILD_TOOL) build -t $(IMAGE_DEEPEVAL) -f Containerfile .
	@echo "✅ Built: $(IMAGE_DEEPEVAL)"

.PHONY: images
images: image-lighteval image-guidellm image-mteb image-deepeval
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

.PHONY: push-deepeval
push-deepeval:
	@echo "Pushing DeepEval adapter image..."
	$(BUILD_TOOL) push $(IMAGE_DEEPEVAL)
	@echo "✅ Pushed: $(IMAGE_DEEPEVAL)"

.PHONY: push-images
push-images: push-lighteval push-guidellm push-mteb push-deepeval
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

.PHONY: clean-deepeval
clean-deepeval:
	@echo "Removing DeepEval adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_DEEPEVAL) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_DEEPEVAL)"

.PHONY: clean-images
clean-images: clean-lighteval clean-guidellm clean-mteb clean-deepeval
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
	@echo "✅ All adapter tests passed"
