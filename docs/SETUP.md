# Setup

## Prerequisites

- Python 3.14 or later (see `requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

```bash
git clone https://github.com/AnikaAchary/mangomusic.git
cd mangomusic
uv sync
```

`uv sync` creates a `.venv` and installs the project (currently no third-party dependencies).

## Running the project

```bash
uv run mangomusic
```

This runs the `mangomusic` entry point (`src/mangomusic/__init__.py:main`), which currently prints `Hello from mangomusic!`.

## Environment variables / config

None required yet — the project has no config files or environment variables at this stage.
