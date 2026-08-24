# Repository layout

```
mangomusic/
├── src/mangomusic/       # package source
│   ├── audio.py          # decoding, resampling, and canonicalization
│   ├── rhythm.py         # onset, tempo, and beat analysis
│   ├── chroma.py         # pitch-class features and beat-synchronous aggregation
│   ├── chords.py         # chord templates and per-beat chord recognition
│   ├── recordings.py     # sample-recording lookup and ground-truth manifest
│   └── errors.py         # public exception hierarchy
├── tests/                # pytest suite, one module per source module
│   ├── test_audio.py
│   ├── test_rhythm.py
│   ├── test_chroma.py
│   ├── test_chords.py
│   ├── test_recordings.py
│   ├── test_recording_samples.py   # skipped unless the recordings are present
│   └── data/             # ground truth for the sample recordings
│       └── recordings.json
├── notebooks/            # exploratory notebooks; outputs stripped on commit
│   └── chromagram_review.ipynb
├── docs/                 # this documentation
│   ├── architecture.md
│   ├── contributing.md
│   ├── repo-layout.md
│   └── pipeline/         # one page per pipeline stage
├── .gitattributes        # routes notebooks through the nbstripout filter
├── AGENTS.md             # instructions for coding agents working in this repo
├── CLAUDE.md             # points at AGENTS.md
├── pyproject.toml        # project metadata, dependencies, and tool config
├── uv.lock               # committed lockfile
└── README.md
```

## Where things belong

**`src/` layout.** The package lives under `src/` rather than at the repository
root, so tests import the installed package rather than accidentally importing
the source tree from the working directory. `uv sync` installs the project in
editable mode, so source edits take effect without reinstalling.

**One module per pipeline stage.** Each module in `src/mangomusic/` is one stage
of the pipeline described in [architecture.md](architecture.md), and each is a
set of plain functions plus the pydantic models describing its results. Stages
depend downward only: `chords` imports from `chroma`, and nothing imports from
`chords`.

**`errors.py` holds the public exception hierarchy.** `MangoMusicError` is the
base class for expected failures, so callers can catch everything MangoMusic
raises deliberately with one `except`. `ValueError` for malformed arguments is
raised directly and is deliberately not part of that hierarchy — see
[architecture.md](architecture.md#empty-and-degenerate-input).

**Tests mirror the package.** `tests/test_<module>.py` for
`src/mangomusic/<module>.py`. Fixtures are generated signals; audio files are
never committed. See [contributing.md](contributing.md).

`tests/test_recording_samples.py` is the one exception to that mirroring: it is
named for the data it checks rather than a module, because it validates real
recordings across several modules at once. Those recordings live outside the
repository, so it skips unless they are present — see
[contributing.md](contributing.md#sample-recordings).

**`notebooks/` is for exploration, not for checks.** Notebooks are how you look
at a recording and decide what is in it; the assertions that must keep passing
belong in `tests/`. `.gitattributes` routes every notebook through nbstripout so
outputs never enter git.

**Documentation.** `README.md` is the overview and the entry point: status,
install, quickstart, roadmap, and links here. `docs/` holds the detail, one page
per stage plus the cross-cutting pages. `AGENTS.md` is the contract for coding
agents and the authoritative statement of the workflow rules.

## Configuration

`pyproject.toml` holds everything: project metadata, runtime dependencies, the
dev dependency group, and tool configuration for Ruff and pyright. There are no
separate tool config files, and none should be added — Ruff replaces black,
isort, and flake8 in this project.

Dependencies are managed with `uv add` / `uv remove`, never by hand-editing
`pyproject.toml`, so `uv.lock` stays consistent. The lockfile is committed and
belongs in any change that touches dependencies.
