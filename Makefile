.PHONY: check fmt fmt-check lint typecheck test test-fast test-model test-model-slow dev sync clean sec ci

check: fmt lint typecheck test

fmt:
	uv run ruff format src/ tests/

fmt-check:
	uv run ruff format --check src/ tests/

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

sec:
	uv run pip-audit
	uv run bandit -c pyproject.toml -r src/

ci: fmt-check lint typecheck test-fast sec

sync:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf results/
	rm -rf .*.nc
