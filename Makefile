# Makefile for py-statmatch development

.PHONY: help install format lint test test-cov documentation documentation-build documentation-serve documentation-dev changelog build clean

help:
	@echo "Available commands:"
	@echo "  make install          Install package with development dependencies"
	@echo "  make format           Format code with Black and isort"
	@echo "  make lint             Check code style with Black and flake8"
	@echo "  make test             Run test suite"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make documentation    Build documentation"
	@echo "  make documentation-serve  Serve documentation locally"
	@echo "  make changelog        Update changelog from changelog_entry.yaml"
	@echo "  make build            Build distribution packages"
	@echo "  make clean            Clean build artifacts"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

format:
	black . -l 79
	isort . --profile black --line-length 79

lint:
	black . -l 79 --check
	isort . --profile black --line-length 79 --check-only
	flake8 statmatch tests --max-line-length 79 --extend-ignore E203,W503

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=statmatch --cov-report=term-missing --cov-report=html --cov-report=xml

test-types:
	mypy statmatch --strict

test-all: lint test-types test-cov

documentation:
	rm -rf docs/_build docs/.jupyter_cache
	cd docs && jupyter-book build . --all

documentation-build: documentation

documentation-serve: documentation
	cd docs && python -m http.server -d _build/html 8000

documentation-dev:
	cd docs && jupyter-book build . --all
	cd docs && python -m http.server -d _build/html 8000

changelog:
	@if [ -f changelog_entry.yaml ]; then \
		yaml-changelog . --release; \
		echo "Changelog updated. Remember to remove changelog_entry.yaml"; \
	else \
		echo "No changelog_entry.yaml found"; \
	fi

build: clean
	python -m build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf docs/_build docs/.jupyter_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow commands
dev-setup: install
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make format' to format code"
	@echo "Run 'make documentation-serve' to view docs"

# CI simulation commands
ci-local: format lint test-cov
	@echo "All CI checks passed locally!"

# Version management
bump-version:
	@if [ -f changelog_entry.yaml ]; then \
		yaml-changelog . --release; \
		VERSION=$$(grep -E "^version:" changelog.yaml | head -1 | cut -d' ' -f2); \
		sed -i.bak "s/^version = .*/version = \"$$VERSION\"/" pyproject.toml; \
		sed -i.bak "s/__version__ = .*/__version__ = \"$$VERSION\"/" statmatch/__init__.py; \
		rm -f pyproject.toml.bak statmatch/__init__.py.bak; \
		rm -f changelog_entry.yaml; \
		echo "Version bumped to $$VERSION"; \
	else \
		echo "No changelog_entry.yaml found"; \
		exit 1; \
	fi