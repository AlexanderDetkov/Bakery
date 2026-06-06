.PHONY: install test test-fast smoke lint

install:
	pip install -e ".[dev,analysis]"

test:
	pytest

# Everything except the model-loading end-to-end smoke (used by the research loop's gate).
test-fast:
	pytest -m "not slow"

# Tiny end-to-end bake on a stub model, CPU, no gating. Verifies the whole pipeline.
smoke:
	python run.py --experiment bake_smoke --model.device cpu

lint:
	ruff check bakery tests
