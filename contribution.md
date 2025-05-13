# Contribution Guidelines

## Environment Setup

### Install uv

[uv](https://github.com/astral.sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Install Dependencies

```bash
# Install dependencies from pyproject.toml
uv sync
```

### Install Pre-commit Hooks

We use pre-commit to ensure code quality and consistency.

```bash
# Install pre-commit hooks
uv run pre-commit install
```

## Working with Marimo Notebooks

### Creating a New Notebook

Create a new notebook using marimo:

```bash
uv run marimo edit notebook.py --sandbox
```

### Publishing Notebooks

Add the following workflow to `.github/workflows/publish-marimo.yml`:

```yaml
...
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      ...
      - name: Export notebook
        run: |
          uv run marimo export html notebook.py -o build/notebook.html --sandbox
      ...
```

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request with a clear description of changes