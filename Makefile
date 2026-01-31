# EvalHub Adapters Makefile
# Build container images for various evaluation framework adapters

# Variables
REGISTRY ?= quay.io/eval-hub
BUILD_TOOL ?= podman
VERSION ?= latest

# Image names
IMAGE_LIGHTEVAL = $(REGISTRY)/community-lighteval:$(VERSION)

# Default target
.PHONY: help
help:
	@echo "EvalHub Adapters Build Targets"
	@echo "==============================="
	@echo ""
	@echo "Image Build:"
	@echo "  make image-lighteval    - Build LightEval adapter image"
	@echo "  make images             - Build all adapter images"
	@echo ""
	@echo "Image Push:"
	@echo "  make push-lighteval     - Push LightEval adapter image"
	@echo "  make push-images        - Push all adapter images"
	@echo ""
	@echo "Clean:"
	@echo "  make clean-lighteval    - Remove LightEval adapter image"
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

.PHONY: images
images: image-lighteval
	@echo "✅ All adapter images built"

# Push targets
.PHONY: push-lighteval
push-lighteval:
	@echo "Pushing LightEval adapter image..."
	$(BUILD_TOOL) push $(IMAGE_LIGHTEVAL)
	@echo "✅ Pushed: $(IMAGE_LIGHTEVAL)"

.PHONY: push-images
push-images: push-lighteval
	@echo "✅ All adapter images pushed"

# Clean targets
.PHONY: clean-lighteval
clean-lighteval:
	@echo "Removing LightEval adapter image..."
	$(BUILD_TOOL) rmi $(IMAGE_LIGHTEVAL) 2>/dev/null || true
	@echo "✅ Removed: $(IMAGE_LIGHTEVAL)"

.PHONY: clean-images
clean-images: clean-lighteval
	@echo "✅ All adapter images removed"

# Development targets
.PHONY: build-and-push-lighteval
build-and-push-lighteval: image-lighteval push-lighteval
	@echo "✅ LightEval adapter built and pushed"

.PHONY: build-and-push-all
build-and-push-all: images push-images
	@echo "✅ All adapters built and pushed"
