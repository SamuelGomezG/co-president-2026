.PHONY: check fmt fmt-check lint typecheck test test-fast test-model test-model-slow dev sync install clean sec ci download-cnpv cnpv nbi ipm

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
	uv run pytest tests/ -v --cov=src/co_president --cov-report=term-missing --cov-report=xml

test-fast:
	uv run pytest tests/ -v --ignore=tests/test_model.py --ignore=tests/integration

test-model:
	uv run pytest tests/test_model.py -v -m "not slow"

test-model-slow:
	uv run pytest tests/test_model.py -v -m "slow"

dev:
	@if [ -f src/co_president/__main__.py ]; then \
		uv run python -m co_president run --no-sample; \
	else \
		echo "dev: __main__.py not yet implemented (SPEC-10 pending)"; \
	fi

sec:
	uv run pip-audit --skip-editable
	uv run bandit -c pyproject.toml -r src/

ci: fmt-check lint typecheck test-fast sec

sync:
	uv sync --extra dev

install:
	uv pip install -e .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .tox -exec rm -rf {} + 2>/dev/null || true
	rm -rf results/
	rm -rf .*.nc
	rm -rf .coverage coverage.xml htmlcov/
	rm -rf dist/ build/

download-cnpv:
	uv run python scripts/download_cnpv_2018.py

CNPV_ZIPS := $(wildcard data/cnpv-2018/raw/*.zip)
CNPV_OUT := data/fundamentals/cnpv_2018.csv

cnpv: $(CNPV_OUT)

$(CNPV_OUT): $(CNPV_ZIPS) src/co_president/ingestion/ingest_cnpv.py
	uv run python -m co_president ingest --component cnpv

NBI_SRC := data/raw/DANE-NBI/CNPV-2018-NBI.xlsx
NBI_OUT := data/fundamentals/nbi_2018.csv

nbi: $(NBI_OUT)

$(NBI_OUT): $(NBI_SRC) src/co_president/ingestion/ingest_nbi.py
	uv run python -m co_president ingest --component nbi

IPM_ZIPS := $(wildcard data/raw/IPM-2018/Hogares*.zip data/raw/IPM-2022/hogares*.zip)
IPM_OUT := data/fundamentals/ipm_2018.csv

ipm: $(IPM_OUT)

$(IPM_OUT): $(IPM_ZIPS) src/co_president/ingestion/ingest_ipm.py
	uv run python -m co_president ingest --component ipm
