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

That creates the virtual environment and installs both runtime and dev dependencies. Run anything in the project with `uv run`.

The pipeline is not yet wrapped in a command, but the stages that exist compose end to end — audio file in, per-beat chord labels out:

```python
from pathlib import Path

import numpy as np

from mangomusic.audio import load_audio
from mangomusic.chords import recognize_chords
from mangomusic.chroma import aggregate_chroma_by_beat, compute_chroma
from mangomusic.rhythm import analyze_rhythm

samples, sample_rate_hz = load_audio(Path("song.mp3"), 22_050)

rhythm = analyze_rhythm(samples, sample_rate_hz)
beat_times_seconds = np.array(
    [beat.timestamp_seconds for beat in rhythm.beats],
    dtype=np.float64,
)

chroma, frame_times_seconds = compute_chroma(samples, sample_rate_hz)
beat_chroma = aggregate_chroma_by_beat(chroma, frame_times_seconds, beat_times_seconds)

analysis = recognize_chords(beat_chroma, beat_times_seconds)
for label in analysis.labels:
    print(f"{label.timestamp_seconds:6.2f}  {label.symbol}  {label.confidence:.2f}")
```

Audio with no detectable beat produces an empty beat grid, which `recognize_chords` rejects — guard on `rhythm.beats` before recognizing. Chart generation is still under development.

## Documentation

[`docs/`](docs/README.md) has the detail.

- [Architecture](docs/architecture.md) — how the stages compose, and the conventions they share
- [Repository layout](docs/repo-layout.md) — where code, tests, and docs live
- [Contributing](docs/contributing.md) — quality gates, testing, and git workflow

Per stage, in pipeline order:

- [Audio ingestion](docs/pipeline/audio.md) — decoding a file to mono `float32` samples
- [Rhythm analysis](docs/pipeline/rhythm.md) — onset strength, tempo, and beat tracking
- [Chroma features](docs/pipeline/chroma.md) — pitch-class energy, per frame and per beat
- [Chord recognition](docs/pipeline/chords.md) — scoring beats against chord templates

## Development

Four quality gates, all runnable through `uv`:

```shell
uv run ruff format .      # formatting
uv run ruff check .       # linting
uv run pyright            # type checking
uv run pytest             # tests
```

All four must pass before a change is considered done. [`AGENTS.md`](AGENTS.md) has the full definition-of-done gate, git workflow, and prohibited patterns — read it before contributing, and point any coding agent at it too. [`docs/contributing.md`](docs/contributing.md) is the same material in readable form.

## Roadmap

- [x] Audio loading and resampling
- [x] Onset strength, global tempo, and beat tracking
- [ ] Downbeat inference
- [x] Chroma feature extraction and beat-synchronous aggregation
- [x] Chord recognition
- [ ] Beat-aligned segmentation
- [ ] Chord sheet rendering

## Contributing

Work happens on branches with pull requests. Run the quality gates before opening one. Details in [`docs/contributing.md`](docs/contributing.md) and [`AGENTS.md`](AGENTS.md).
