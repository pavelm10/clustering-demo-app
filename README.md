# clustering-demo-app

This repository implements various clustering algorithms using scikit-learn library.

# Installation

## Prerequisites

- poetry (>=2.0.0)
- python 3.12
- pre-commit package installed, only for development purposes

## Installation steps

To install the virtual environment run:
```
poetry install
```

# Development

The repository is developed in Python and `ruff` and `ty` are used for formatting and
type checking respectively.

Please install pre-commit with:
```
pre-commit install
```

## Tests

Run tests with:
```
poetry run pytest tests/
```
