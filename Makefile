# --- Make Targets ---

# Core Build Targets
.PHONY: all
all: images

# --- BUILD TARGETS ---
# Builds the container image for one specific adapter.
# Usage: make image-<adapter-name>
image-<adapter-name>: $(adapter-name)
	@echo "Building container image for <adapter-name>..."

# Builds all adapters.
images: $(adapters)

# Builds and pushes all adapters to a specified registry.
# Usage: make push-all REGISTRY=quay.io/myorg
push-all: images
	@if [ -z "$REGISTRY" ]; then echo "Error: Please define REGISTRY target."; exit 1; fi
	@for adapter in $(adapters); do \
		echo "Pushing <adapter> to $REGISTRY"; \
		make push-${adapter} REGISTRY="$REGISTRY"; \
	done

# Builds and pushes a single adapter.
# Usage: make push-<adapter-name> REGISTRY=quay.io/myorg
push-<adapter-name>: images
	@poetry run python -m build -s -w -t wheel . # Assuming standard Python packaging process
	@# podman build -t ${REGISTRY}/$<adapter-name>:latest ./adapters/<adapter-name>
	@echo "MOCK: Pushing $<adapter-name> image to $REGISTRY/$<adapter-name>:latest"


# --- TEST TARGETS ---
.PHONY: tests
tests: tests-all

# Runs tests for all registered adapters.
tests-all: $(adapters)
	@echo "Running all adapter tests..."
	@for adapter in $(adapters); do \
		echo "============================================="; \
		echo "Running tests for $adapter..."; \
		make test-$adapter; \
	done

# Runs tests for a single adapter.
# Usage: make test-<adapter-name>
test-<adapter-name>:
	@if [ ! -d "adapters/<adapter-name>" ]; then echo "Error: Adapter directory not found for <adapter-name>."; exit 1; fi
	@echo "Running pytest for <adapter-name> in $(pwd)/adapters/<adapter-name>"; \
	@# pytest adapters/<adapter-name>/tests/ --cov=adapters/<adapter-name>/src \
	@# Note: Requires virtual environment setup (see adapter's README)
	@echo "MOCK: Pytest execution for <adapter-name> complete. Check venv setup."

# --- ADAPTER DEFINITIONS ---
adapters:
	@echo "Registered adapters: lighteval, guidellm, clear, mteb, ruler"

# --- RULER SPECIFIC TARGETS ---
image-ruler-adapter: ruler
test-ruler-adapter: ruler

# =================================================================
# List of all managed adapters
# =================================================================
adapters: lighteval guidellm clear mteb ruler

