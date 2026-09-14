# Contributing to Tensorlink

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.3+
- Git

### Getting Started

1. Fork and clone the repository:
```bash
git clone https://github.com/mattjhawken/tensorlink.git
cd tensorlink
```

2. Install Poetry if you don't have it:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:
```bash
poetry --version
```

3. Install all dependencies. This also installs Tensorlink in editable mode:
```bash
poetry install --with dev
```

4. Now you may run Python commands through Poetry. For example:

```bash
poetry run pytest
poetry run python bin/run_node.py
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

Run the full test suite. Adding `-s` will enable node logging to show in real time:
```bash
pytest -s
```

### Code Quality

Before committing, run:
```bash
pre-commit run -a
```

This runs:
- Black (code formatting)
- Flake8 (linting)
- isort (import sorting)
- Type checking (if applicable)

### Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run pre-commit: `pre-commit run -a`
5. Commit with clear messages
6. Push and create a PR

---

## Creating a New Release

Tensorlink uses [Poetry](https://python-poetry.org/) for packaging and follows
[PEP 440](https://peps.python.org/pep-0440/) versioning (`MAJOR.MINOR.PATCH`,
e.g. `0.3.1`). Releases are triggered by pushing a git tag. The CI workflow
handles building, publishing to PyPI, and creating the GitHub Release with
attached artifacts automatically.


### Release steps
 
**1. Bump the version in `pyproject.toml`**
 
Use Poetry's built-in version command so it updates the file correctly:
 
```bash
# Explicit version
poetry version 0.3.1
 
# Or let Poetry increment for you
poetry version patch   # 0.3.0 → 0.3.1
poetry version minor   # 0.3.0 → 0.4.0
poetry version major   # 0.3.0 → 1.0.0
```
 
**2. Regenerate the lockfile**
 
```bash
poetry lock
```
 
**3. Commit and merge via PR**
 
```bash
git checkout -b release/0.3.1
git add pyproject.toml poetry.lock
git commit -m "chore: bump version to 0.3.1"
git push origin release/0.3.1
# Open a PR, get it reviewed, and merge to main
```
 
**4. Tag the release**
 
After the PR is merged, pull main and push the tag:
 
```bash
git checkout main
git pull origin main
git tag v0.3.1
git push origin v0.3.1
```
 
> ⚠️ The tag **must match** the version in `pyproject.toml` exactly (minus the
> `v` prefix). The release workflow verifies this and will fail early if they
> don't match — nothing will be published until they're in sync.
 
---

## Questions?

Join our [Discord](https://discord.gg/aCW2kTNzJ2) for development discussions!
