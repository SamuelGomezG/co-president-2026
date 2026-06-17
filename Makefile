.DEFAULT_GOAL := help
.PHONY: help check fmt fmt-check lint lint-fix typecheck test test-fast test-model test-model-slow \
        dev sync install setup clean sec ci download-cnpv fundamentals cnpv nbi ipm population \
        benchmark-fnn-clr

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check: fmt lint typecheck test  ## Run all quality gates (fmt → lint → typecheck → test)

fmt:  ## Format code with ruff
	@uv run ruff format src/ tests/

fmt-check:  ## Check formatting without modifying files
	@uv run ruff format --check src/ tests/

lint:  ## Lint with ruff (all rules, zero tolerance)
	@uv run ruff check src/ tests/

lint-fix:  ## Auto-fix lint issues (unused imports, refactors, etc.)
	@uv run ruff check --fix src/ tests/

typecheck:  ## Type-check with pyright (strict mode, src/ only)
	@NODE_OPTIONS="--max-old-space-size=4096" uv run pyright src/

test:  ## Run all tests with coverage
	@uv run pytest tests/ -v -n auto --cov=src/co_president --cov-report=term-missing --cov-report=xml

test-fast:  ## Run tests excluding slow MCMC model tests
	@uv run pytest tests/ -v -n auto --ignore=tests/test_model.py --ignore=tests/integration

test-model:  ## Run fast model tests only (graph + prior predictive)
	@uv run pytest tests/test_model.py -v -m "not slow"

test-model-slow:  ## Run slow MCMC tests only (convergence + sanity)
	@uv run pytest tests/test_model.py -v -m "slow"

dev:  ## Run the dev entrypoint (if implemented)
	@if [ -f src/co_president/__main__.py ]; then \
		uv run python -m co_president run --no-sample; \
	else \
		echo "dev: __main__.py not yet implemented (SPEC-10 pending)"; \
	fi

sec:  ## Run security scans (pip-audit + bandit)
	@uv run pip-audit --skip-editable --ignore-vuln CVE-2025-3000 --ignore-vuln PYSEC-2026-196
	@uv run bandit -c pyproject.toml -r src/

ci: fmt-check lint typecheck test-fast sec  ## CI gate (fmt-check → lint → typecheck → test-fast → sec)

sync:  ## Install runtime + dev dependencies (uv sync)
	@uv sync --extra dev

install:  ## Install co-president in editable mode
	@uv pip install -e .

setup:  ## First-time setup: sync dependencies + install local package
	@$(MAKE) sync
	@$(MAKE) install

clean:  ## Remove all caches, build artifacts, and coverage data
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .tox -exec rm -rf {} + 2>/dev/null || true
	@rm -rf results/
	@rm -rf .*.nc
	@rm -rf .coverage coverage.xml htmlcov/
	@rm -rf dist/ build/

download-cnpv:  ## Download raw CNPV 2018 ZIP archives
	@uv run python scripts/download_cnpv_2018.py

benchmark-fnn-clr:  ## Run SPEC-41 FNN+CLR ML benchmark (15 model-transform combinations)
	@uv run python -m co_president benchmark-fnn-clr

fundamentals: cnpv nbi ipm population  ## Ingest all fundamental datasets

CNPV_ZIPS := $(wildcard data/cnpv-2018/raw/*.zip)
CNPV_OUT := data/fundamentals/cnpv_2018.csv

cnpv: $(CNPV_OUT)  ## Ingest CNPV 2018 census data

$(CNPV_OUT): $(CNPV_ZIPS) src/co_president/ingestion/ingest_cnpv.py
	@uv run python -m co_president ingest --component cnpv

NBI_SRC := data/raw/DANE-NBI/CNPV-2018-NBI.xlsx
NBI_OUT := data/fundamentals/nbi_2018.csv

nbi: $(NBI_OUT)  ## Ingest NBI (unsatisfied basic needs) data

$(NBI_OUT): $(NBI_SRC) src/co_president/ingestion/ingest_nbi.py
	@uv run python -m co_president ingest --component nbi

IPM_ZIPS := $(wildcard data/raw/IPM-2018/Hogares*.zip data/raw/IPM-2022/hogares*.zip)
IPM_OUT := data/fundamentals/ipm_2018.csv

ipm: $(IPM_OUT)  ## Ingest IPM (multidimensional poverty) data

$(IPM_OUT): $(IPM_ZIPS) src/co_president/ingestion/ingest_ipm.py
	@uv run python -m co_president ingest --component ipm

POPULATION_SRC := data/raw/PPED-AreaMun-2018-2042_VP.xlsx
POPULATION_OUT := data/fundamentals/population_2018_2026.csv

population: $(POPULATION_OUT)  ## Ingest population projections data

$(POPULATION_OUT): $(POPULATION_SRC) src/co_president/ingestion/ingest_population.py
	@uv run python -m co_president ingest --component population
