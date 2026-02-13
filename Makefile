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
	@echo "  make images             - Build all adapter images"
	@echo ""
	@echo "Image Push:"
	@echo "  make push-lighteval     - Push LightEval adapter image"
	@echo "  make push-guidellm      - Push GuideLLM adapter image"
	@echo "  make push-mteb          - Push MTEB adapter image"
	@echo "  make push-images        - Push all adapter images"
	@echo ""
	@echo "Clean:"
	@echo "  make clean-lighteval    - Remove LightEval adapter image"
	@echo "  make clean-guidellm     - Remove GuideLLM adapter image"
	@echo "  make clean-mteb         - Remove MTEB adapter image"
	@echo "  make clean-images       - Remove all adapter images"
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

.PHONY: images
images: image-lighteval image-guidellm image-mteb
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

.PHONY: push-images
push-images: push-lighteval push-guidellm push-mteb
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

.PHONY: clean-images
clean-images: clean-lighteval clean-guidellm clean-mteb
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

.PHONY: build-and-push-all
build-and-push-all: images push-images
	@echo "✅ All adapters built and pushed"
