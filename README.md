# mangomusic

Automatic music transcription: audio in, chord sheet out.

`mangomusic` takes a recording and produces a readable chord chart — the sequence of chords, when each one lands, and how they line up against the song's beat and bar structure.

> **Status: early development.** The scaffolding, tooling, and conventions are in place; the transcription pipeline is being built out. Interfaces will change without warning.

## Requirements

- Python (version pinned in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- FFmpeg, for decoding compressed audio formats

## Quickstart

```shell
git clone <repo-url>
cd mangomusic
uv sync
```

That creates the virtual environment and installs both runtime and dev dependencies. Run anything in the project with `uv run`:

## Project layout

```
mangomusic/
├── src/mangomusic/       # package source
│   └── models.py         # MangoModel — shared Pydantic base class
├── data/                 # Data and music files
├── tests/                # pytest suite
├── docs/
├── AGENTS.md             # instructions for coding agents working in this repo
├── pyproject.toml
└── README.md
```

## Development

Three quality gates, all runnable through `uv`:

```shell
uv run ruff format .      # formatting
uv run ruff check .       # linting
uv run pyright            # type checking
uv run pytest             # tests
```

All four must pass before a change is considered done. `AGENTS.md` has the full definition-of-done gate, git workflow, and prohibited patterns — read it before contributing, and point any coding agent at it too.

## Roadmap

- [ ] Audio loading and resampling
- [ ] Beat and downbeat tracking
- [ ] Chroma feature extraction
- [ ] Chord recognition
- [ ] Beat-aligned segmentation
- [ ] Chord sheet rendering

## Contributing

Work happens on branches with pull requests. Run the quality gates before opening one. Details in `AGENTS.md`.
