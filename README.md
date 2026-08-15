# mangomusic

Automatic music transcription: audio in, chord sheet out.

`mangomusic` takes a recording and produces a readable chord chart — the sequence of chords, when each one lands, and how they line up against the song's beat and bar structure.

> **Status: early development.** The scaffolding, tooling, and conventions are in place; the transcription pipeline is being built out. Interfaces will change without warning.

## Requirements

- Python (version pinned in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [FFmpeg](https://ffmpeg.org/), available on `PATH`, for decoding audio

## Quickstart

```shell
git clone <repo-url>
cd mangomusic
uv sync
```

That creates the virtual environment and installs both runtime and dev dependencies. Run anything in the project with `uv run`:

## Audio ingestion

MangoMusic can currently decode an FFmpeg-supported audio file into a mono
`float32` NumPy signal at a requested sample rate:

```python
from pathlib import Path

from mangomusic.audio import load_audio

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
```

Chroma, chord recognition, and chart generation are still under development.

## Rhythm analysis

MangoMusic can compute an onset-strength representation and analyze decoded
audio for a global tempo, ordered beat timestamps, and confidence estimates:

```python
from pathlib import Path

from mangomusic.audio import load_audio
from mangomusic.rhythm import analyze_rhythm, compute_onset_strength

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)
onset_strength, onset_times_seconds = compute_onset_strength(
    samples,
    sample_rate_hz,
)
rhythm = analyze_rhythm(samples, sample_rate_hz)
```

Beat grouping assumes 4/4 time. The first detected beat is labeled beat one of
bar one; this initial implementation does not infer the musical downbeat.
Silent and non-rhythmic inputs return no BPM or beats and zero tempo confidence.

## Project layout

```
mangomusic/
├── src/mangomusic/       # package source
│   ├── audio.py          # decoding, resampling, and canonicalization
│   ├── rhythm.py         # onset, tempo, and beat analysis
│   └── errors.py         # public exception hierarchy
├── tests/                # pytest suite
├── docs/
├── AGENTS.md             # instructions for coding agents working in this repo
├── pyproject.toml
└── README.md
```

## Development

Four quality gates, all runnable through `uv`:

```shell
uv run ruff format .      # formatting
uv run ruff check .       # linting
uv run pyright            # type checking
uv run pytest             # tests
```

All four must pass before a change is considered done. `AGENTS.md` has the full definition-of-done gate, git workflow, and prohibited patterns — read it before contributing, and point any coding agent at it too.

## Roadmap

- [ ] Audio loading and resampling
- [x] Onset strength, global tempo, and beat tracking
- [ ] Downbeat inference
- [ ] Chroma feature extraction
- [ ] Chord recognition
- [ ] Beat-aligned segmentation
- [ ] Chord sheet rendering

## Contributing

Work happens on branches with pull requests. Run the quality gates before opening one. Details in `AGENTS.md`.
