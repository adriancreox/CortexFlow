.PHONY: install dev test lint type bench clean docs

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -x -q --no-cov

lint:
	ruff check cortexflow tests examples
	ruff format --check cortexflow tests examples

format:
	ruff format cortexflow tests examples
	ruff check --fix cortexflow tests examples

type:
	mypy cortexflow

bench:
	python -m pytest tests/bench/ -v --benchmark-only

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info

ci: lint type test

hello:
	python examples/hello_agent.py

pipeline:
	python examples/multi_agent_pipeline.py
