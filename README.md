# Nexus 5a

A DSPy-based project for question answering using ReAct and optimization.

## Installation

This project uses Poetry for dependency management. To install dependencies:

```bash
poetry install
```

## Usage

To run the main script:

```bash
poetry run python -m nexus_5a.main
```

## Development

To run tests:

```bash
poetry run pytest
```

To format code:

```bash
poetry run black .
poetry run isort .
```

To run type checking:

```bash
poetry run mypy src/
``` 