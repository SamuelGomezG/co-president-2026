.PHONY: check fmt lint typecheck test test-fast test-model test-model-slow dev sync clean

check: fmt lint typecheck test

fmt:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run pyright src/

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v --ignore=tests/test_model.py

test-model:
	uv run pytest tests/test_model.py -v -m "not slow"

test-model-slow:
	uv run pytest tests/test_model.py -v -m "slow"

dev:
	uv run python -m co_president run --no-sample

sync:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf results/
	rm -rf .*.nc
