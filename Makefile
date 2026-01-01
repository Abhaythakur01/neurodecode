.PHONY: help install install-dev clean test test-unit test-integration test-all lint format docker-build docker-up docker-down

help:
	@echo "NeuroDecode Development Commands"
	@echo "================================="
	@echo "install          Install package and dependencies"
	@echo "install-dev      Install development dependencies"
	@echo "clean            Clean build artifacts"
	@echo "test             Run all tests"
	@echo "test-unit        Run unit tests only"
	@echo "test-integration Run integration tests only"
	@echo "test-coverage    Run tests with coverage report"
	@echo "lint             Run linters"
	@echo "format           Format code with black and isort"
	@echo "docker-build     Build Docker images"
	@echo "docker-up        Start Docker services"
	@echo "docker-down      Stop Docker services"
	@echo "docs             Build documentation"
	@echo "pre-commit       Install pre-commit hooks"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v -m "not slow"

test-integration:
	pytest tests/integration -v -m integration

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests --max-line-length=100 --extend-ignore=E203,W503
	mypy src --ignore-missing-imports

format:
	black src tests
	isort src tests

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docs:
	cd docs && make html

pre-commit:
	pre-commit install
	pre-commit run --all-files

benchmark:
	pytest tests/benchmarks -v --benchmark-only

.DEFAULT_GOAL := help
