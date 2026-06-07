.PHONY: install test test-fast smoke smoke-reg lint

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

# Like smoke, but exercises the regularization path (anchor trajectories + the behavior_drift metric)
# on the offline synthetic source: generate -> mix -> gate -> drift, end-to-end on CPU, no network.
smoke-reg:
	python run.py --experiment bake_smoke --model.device cpu \
	  --regularization.num_train_contexts 2 --regularization.eval_num_contexts 2 \
	  --regularization.source synthetic --regularization.context_split train \
	  --regularization.max_new_tokens 4 --eval.metrics eval_kl,behavior_drift

lint:
	ruff check bakery tests
