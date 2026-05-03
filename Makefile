.PHONY: install install-dev test test-fast lint format typecheck clean help

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
RUFF ?= ruff

# ── Default ───────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────────────

install: ## Install package and core dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install-dev: ## Install package with dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]" --no-deps
	pre-commit install || true

# ── Test ──────────────────────────────────────────────────────────────────────

test: ## Run all tests (excluding slow/gpu)
	$(PYTEST) tests/ -v --tb=short -x -m "not slow and not gpu"

test-all: ## Run all tests including slow
	$(PYTEST) tests/ -v --tb=short

test-fast: ## Run tests in parallel
	$(PYTEST) tests/ -v --tb=short -x -m "not slow and not gpu" -n auto

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --tb=short -x -m "not slow and not gpu" \
		--cov=vl_jepa --cov-report=term-missing --cov-report=html

test-smoke: ## Run only smoke tests
	$(PYTEST) tests/test_smoke.py -v --tb=short

# ── Lint / Format ─────────────────────────────────────────────────────────────

lint: ## Run linter (ruff)
	$(RUFF) check vl_jepa/ tests/

format: ## Auto-format code (ruff + isort)
	$(RUFF) format vl_jepa/ tests/
	$(RUFF) check --fix vl_jepa/ tests/

typecheck: ## Run type checking (mypy)
	mypy vl_jepa/ --ignore-missing-imports

# ── Clean ─────────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf test-results.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	@echo "Clean."

clean-all: clean ## Remove everything including venv and checkpoints
	rm -rf .venv/ venv/ env/
	rm -rf checkpoints/ outputs/ logs/ wandb/
	@echo "Deep clean."
